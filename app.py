import onnxruntime_genai as og
import time
from fastapi import FastAPI, Query, Form, HTTPException
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
    prompt: str = Form(..., description="Input prompt to generate tokens from"),
    rules: str = Form(default="You are a digital assistant", description="Rules for token generation", min_length=1),
    max_length: Optional[int] = Query(1000, description="Max number of tokens to generate including the prompt"),
    min_length: Optional[int] = Query(10, description="Min number of tokens to generate including the prompt"),
    top_p: Optional[float] = Query(default=0.5, description="Top p probability to sample with"),
    top_k: Optional[int] = Query(default=100, description="Top k tokens to sample from"),
    temperature: Optional[float] = Query(default=0.5, description="Temperature to sample with"),
    repetition_penalty: Optional[float] = Query(None, description="Repetition penalty to sample with"),
    do_random_sampling: Optional[bool] = Query(False, description="Do random sampling"),
    batch_size_for_cuda_graph: Optional[int] = Query(1, description="Max batch size for CUDA graph"),
    verbose: Optional[bool] = Query(True, description="Print verbose output and timing information")
):
    if verbose: print("Starting token generation process...")

    # Apply the fixed chat template with rules and prompt
    chat_template = f"<|system|>\n{rules}<|end>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"

    # Tokenize prompts
    input_tokens = tokenizer.encode_batch([chat_template])
    if verbose: print(f'Prompt(s) encoded: {chat_template}')

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
    params.try_graph_capture_with_max_batch_size(1)
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
    generated_text = tokenizer.decode(output_tokens[0]).

    if verbose:
        total_tokens = len(output_tokens[0])
        print(f"Tokens: {total_tokens} Time: {run_time:.2f} Tokens per second: {total_tokens/run_time:.2f}")

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens": len(output_tokens[0]),
        "time": run_time,
        "tokens_per_second": len(output_tokens[0]) / run_time
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
