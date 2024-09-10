from fastapi import FastAPI, Form
from pydantic import BaseModel
import onnxruntime_genai as og
import time

app = FastAPI()

class GenerateRequest(BaseModel):
    model: str
    input: str
    rules: str
    min_length: int | None = None
    max_length: int | None = None
    do_sample: bool | None = False
    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
    verbose: bool | None = True
    timings: bool | None = True

@app.post("/generate/")
async def generate(
    model: str = Form("models/phi-3-mini-4k-instruct-onnx"),
    input: str = Form(...),
    rules: str = Form("you are a digital assistant"),
    min_length: int = Form(...),
    max_length: int = Form(2000),
    do_sample: bool = Form(False),
    top_p: float = Form(...),
    top_k: int = Form(...),
    temperature: float = Form(...),
    repetition_penalty: float = Form(...),
    verbose: bool = Form(True),
    timings: bool = Form(True)
):
    if verbose:
        print("Loading model...")

    if timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model_instance = og.Model(f'{model}')
    if verbose:
        print("Model loaded")

    tokenizer = og.Tokenizer(model_instance)
    tokenizer_stream = tokenizer.create_stream()
    
    if verbose:
        print("Tokenizer created")
    
    search_options = {
        'do_sample': do_sample,
        'max_length': max_length,  # Default max length to 16000
        'min_length': min_length,
        'top_p': top_p,
        'top_k': top_k,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty
    }


    chat_template = '<|system|>\n{rules}<|end>\n<|user|>\n{input}<|end|>\n   <|assistant|>'

    text = "Hello, how are you?"
    prompt = f'{chat_template.format(input=text, rules=rules)}'
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model_instance)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model_instance, params)

    if verbose:
        print("Generator created")
        print("Running generation loop ...")

    if timings:
        first = True
        new_tokens = []

    result = ""

    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            if timings and first:
                first_token_timestamp = time.time()
                first = False

            new_token = generator.get_next_tokens()[0]
            result += tokenizer_stream.decode(new_token)

            if timings:
                new_tokens.append(new_token)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    del generator

    if timings:
        prompt_time = first_token_timestamp - started_timestamp
        run_time = time.time() - first_token_timestamp
        timing_info = {
            "prompt_length": len(input_tokens),
            "new_tokens": len(new_tokens),
            "time_to_first": f"{prompt_time:.2f}s",
            "prompt_tokens_per_second": f"{len(input_tokens)/prompt_time:.2f} tps",
            "new_tokens_per_second": f"{len(new_tokens)/run_time:.2f} tps"
        }
        return {"input": input, "output": result, "timing_info": timing_info}

    return {"output": result}
