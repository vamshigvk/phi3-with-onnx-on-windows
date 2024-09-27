import os
import argparse
import numpy as np
import pandas as pd
import time
import onnxruntime_genai as og

choices = ["A", "B", "C", "D"]

# Softmax function to calculate probabilities
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax

# Format subject by replacing underscores with spaces
def format_subject(subject):
    return " ".join(subject.split("_"))

# Format a single example for evaluation
def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

# Generate a prompt for the model to evaluate
def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# Function to evaluate the model's performance
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # Get the prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # Tokenize the prompt using the phi-3 tokenizer
        input_tokens = tokenizer.encode_batch([prompt])
        
        # Create GeneratorParams for token generation
        params = og.GeneratorParams(model)
        params.input_ids = input_tokens
        params.set_search_options(max_length=2000, min_length=1, top_p=0.5, top_k=100)

        # Generate the tokens with the phi-3 model
        output_tokens = model.generate(params)

        # Decode the generated output to get the text
        generated_text = tokenizer.decode(output_tokens[0])

        # Evaluate log probabilities for multiple choice options (A, B, C, D)
        lprobs = []
        for ans in answers:
            # Here you need to compute the logprobs from the output
            # Simulating logprobs for simplicity (you'll need a real implementation)
            if ans in generated_text:
                lprobs.append(-1)  # Assuming a token match (logprob closer to 0 means higher likelihood)
            else:
                lprobs.append(-100)  # Assuming no match (logprob far lower)

        # Get the predicted answer
        pred = choices[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        # Check if the predicted answer is correct
        label = test_df.iloc[i, test_df.shape[1] - 1]
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # Calculate accuracy
    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"Average accuracy: {acc:.3f} - {subject}")

    return cors, acc, all_probs

# Main function to run evaluation
def main(args):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # Load the phi-3 model and tokenizer
    model = og.Model('onnx-built-models/phi-3-mini-4k-instruct-onnx-cpu')
    tokenizer = og.Tokenizer(model)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)

        # Save results to CSV
        test_df[f"{subject}_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df[f"{subject}_choice{choice}_probs"] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    main(args)
