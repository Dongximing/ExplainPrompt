

from qa_preprocess import load_and_preprocess_baseline
import pandas as pd
from openai import OpenAI
from math import exp
import numpy as np
# from IPython.display import display, HTML
import os
import tiktoken
from datasets import load_dataset
from utils import calculate_word_scores, calculate_component_scores, strip_tokenizer_prefix, postproces_inferenced
from qa_preprocess import load_and_preprocess
from peturbation import run_peturbation, do_peturbed_reconstruct
import pickle
def create_no_logit_request(custom_id, user_message):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo-0125",
            "messages": [
                {"role": "user", "content": user_message}
            ],

        },
    }

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                                       ))
def infer_openai_without_logits(sentences):
    import json

    requests = [create_no_logit_request(f"request-{i + 1}", msg) for i, msg in enumerate(sentences)]

    file_path = 'api_requests02.jsonl'
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Write each request to the .jsonl file
    with open(file_path, 'w') as file:
        for request in requests:
            # Convert each dictionary to a JSON string
            json_line = json.dumps(request)
            # Write the JSON string to the file followed by a newline
            file.write(json_line + '\n')

    print(f"File saved successfully to {file_path}")
    batch_input_file = client.files.create(
        file=open(f"{file_path}", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    print(batch_input_file_id)

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "no logits job"
        }
    )
    batch_id = client.batches.list().first_id
    import time
    running = True
    while running == True:
        status = client.batches.retrieve(batch_id)
        print(f"Current status: {status}")
        if status.status == 'completed':
            print("Batch processing is complete.")
            running = False
        elif status.status == 'failed':
            print("Batch processing failed.")
            running = False
        else:
            print("Batch still processing. Waiting...")
            time.sleep(10)  # wait for 10 seconds before checking again

    file_response = client.files.content(status.output_file_id)
    response_file = "response_file02.jsonl"
    with open(response_file, 'w') as file:
        file.write(file_response.text)

    all_data = []
    with open('response_file02.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    responses = []


    output_cost = []
    for  data in all_data:
        response = data["response"]["body"]["choices"][0]["message"]["content"]
        responses.append(response)

        output_cost.append(len(encoding.encode(response)))

    return responses, output_cost

def run_initial_inference(start, end, method):
    df = load_and_preprocess_baseline([start, end])
    sentences = df['query']
    responses, output_cost = infer_openai_without_logits(
        sentences)

    data = []
    for ind, example in enumerate(df.select(range(len(df)-1))):

            data.append(
                {'prompt': example['query'], "real_output": responses[ind],
                 "input_cost": output_cost[ind],
                 }
            )

    result = pd.DataFrame(data)

    return result






def main(method):
    start = 1003
    end = start +100
    inference_df = run_initial_inference(start=start, end=end, method=method)
    inference_df.to_pickle(f"{start}_{end}_flesh_baseline_qa_inferenced_df.pkl")
    print("\ndone the inference")

    print("\n done the postprocess")

if __name__ == "__main__":
    main("similarity")
