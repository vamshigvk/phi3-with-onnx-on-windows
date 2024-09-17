import onnxruntime_genai as og
import time
from fastapi import FastAPI, Query, HTTPException, Form
from typing import List, Optional
import uvicorn

app = FastAPI()

# Load the model when the server starts
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading model...")
    model = og.Model('onnx-built-models/phi-3-mini-4k-instruct-onnx-cpu')  # Update the path
    tokenizer = og.Tokenizer(model)
    print("Model loaded and tokenizer created")

@app.post("/generate/")
def generate_tokens(
    prompt: str = Form(..., description="Ask your query here", min_length=1), 
    rules: str = Form(default="You are a digital assistant", description="Rules LLM should follow" ,min_length=1), 
    max_length: Optional[int] = Query(None, description="Max number of tokens to generate including the prompt"),
    min_length: Optional[int] = Query(None, description="Min number of tokens to generate including the prompt"),
    top_p: Optional[float] = Query(None, description="Top p probability to sample with"),
    top_k: Optional[int] = Query(None, description="Top k tokens to sample from"),
    temperature: Optional[float] = Query(None, description="Temperature to sample with"),
    repetition_penalty: Optional[float] = Query(None, description="Repetition penalty to sample with"),
    do_random_sampling: Optional[bool] = Query(False, description="Do random sampling"),
    batch_size_for_cuda_graph: Optional[int] = Query(1, description="Max batch size for CUDA graph"),
    verbose: Optional[bool] = Query(True, description="Print verbose output and timing information")
):
    if verbose: print("Starting token generation process...")

    # Validate chat template
    # if chat_template:
    #     if chat_template.count('{') != 1 or chat_template.count('}') != 1:
    #         raise HTTPException(status_code=400, detail="Chat template must have exactly one pair of curly braces, e.g., '<|user|>\n{input} <|end|>\n<|assistant|>'")

    #     prompt[:] = [chat_template.format(input=text) for text in prompt]

    chat_template = f'<|system|>\n{rules}<|end>\n<|user|>\n{prompt}<|end|>\n   <|assistant|>'

    # Tokenize prompt
    input_tokens = tokenizer.encode_batch(prompt)
    if verbose: print(f'Prompt(s) encoded: {prompt}')

    # Create generator parameters
    params = og.GeneratorParams(model)

    # Set search options if available
    search_options = {
        "do_sample": do_random_sampling,
        "max_length": max_length,
        "min_length": min_length,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty
    }

    # Filter out None values
    search_options = {k: v for k, v in search_options.items() if v is not None}

    if verbose: print(f'Search options: {search_options}')

    params.set_search_options(**search_options)

    # Handle batch size
    params.try_graph_capture_with_max_batch_size(len(prompt))
    if batch_size_for_cuda_graph:
        params.try_graph_capture_with_max_batch_size(batch_size_for_cuda_graph)

    params.input_ids = input_tokens
    if verbose: print("GeneratorParams created")

    # Start token generation
    if verbose: print("Generating tokens ...\n")
    start_time = time.time()
    output_tokens = model.generate(params)
    run_time = time.time() - start_time

    # Decode and return output
    generated_texts = [tokenizer.decode(output) for output in output_tokens]

    if verbose:
        print(f"Generated texts: {generated_texts}")
        total_tokens = sum(len(x) for x in output_tokens)
        print(f"Tokens: {total_tokens} Time: {run_time:.2f} Tokens per second: {total_tokens/run_time:.2f}")

    return {
        "prompt": prompt,
        "generated_texts": generated_texts,
        "tokens": [len(x) for x in output_tokens],
        "time": run_time,
        "tokens_per_second": sum(len(x) for x in output_tokens) / run_time
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
