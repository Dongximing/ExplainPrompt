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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                                       ""))





def create_request(custom_id, user_message):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo-0125",
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "logprobs": True,
            "top_logprobs": 20,

        },
    }
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


def inference_openai(sentences):
    import json

    requests = [create_request(f"request-{i + 1}", msg) for i, msg in enumerate(sentences)]

    file_path = 'api_requests.jsonl'

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
            "description": "nightly eval job"
        }
    )
    batch_id = client.batches.list().first_id
    print(batch_id)
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
    response_file = "response_file.jsonl"
    with open(response_file, 'w') as file:
            file.write(file_response.text)
    import json
    output_cost = 0
    list_top20_logprobs = []
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    all_data = []
    with open('response_file.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    all_data.sort(key=lambda x: x['custom_id'])
    responses = []
    for  data in all_data:
        top_twenty_logprobs = data["response"]["body"]["choices"][0]["logprobs"]
        response = data["response"]["body"]["choices"][0]["message"]["content"]
        responses.append(response)
        print(f"\nResponse: {response}")
        print(f"\nTop 20 logprobs: {top_twenty_logprobs}")
        token_dict = {
            data["token"]: {tp["token"]: tp["logprob"] for tp in data["top_logprobs"]}
            for data in top_twenty_logprobs["content"]
        }
        output_cost += len(encoding.encode(response))
        # print(f"\n token_dict {token_dict}")
        list_top20_logprobs.append(token_dict)







    return list_top20_logprobs, responses[0], output_cost,responses


def infer_openai_without_logits(sentences):
    import json

    requests = [create_no_logit_request(f"request-{i + 1}", msg) for i, msg in enumerate(sentences)]

    file_path = 'api_requests.jsonl'
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
    response_file = "response_file.jsonl"
    with open(response_file, 'w') as file:
        file.write(file_response.text)

    all_data = []
    with open('response_file.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    responses = []


    output_cost = 0
    for  data in all_data:
        response = data["response"]["body"]["choices"][0]["message"]["content"]
        responses.append(response)

        output_cost += len(encoding.encode(response))

    return responses, output_cost


def mask_each_tokens_for_openai(prompt):
    """
    Mask each tokens for OpenAI.

    :param prompt:
    :return:

    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(prompt)
    masked_lists = [tokens[:i] + tokens[i + 1:] for i in range(len(tokens))]
    masked_sentences = []
    total_input = 0
    for masked_list in masked_lists:
        total_input += len(masked_list)
        masked_sentence = encoding.decode(masked_list)
        masked_sentences.append(masked_sentence)
    masked_sentences.insert(0, prompt)
    encoded_prompt = [encoding.decode_single_token_bytes(token) for token in tokens]
    return masked_sentences, encoded_prompt, total_input


def preprocess_baseline(baseline):
    """

    :return:
    """
    if isinstance(baseline, list):
        baseline_list = []
        for k in baseline:
            baseline_list.append(list(k.values()))
    else:
        baseline_list = list(baseline.values())

    return baseline_list


def calculate_importance_score_tokens(baseline, masked_words, encoded_prompt):
    final_attributes = []
    baseline = [{key: np.round(np.exp(value), decimals=5)} for item in baseline for key, value in
                [next(iter(item.items()))]]
    print(f"baseline{baseline}")
    for masked_word in masked_words:
        final_attribute = []
        # print(f"masked_word{masked_word}")
        for i, bl in enumerate(baseline):
            for key in bl:
                if i < len(masked_word) and key in masked_word[i].keys():
                    logit = masked_word[i][key]
                    prob = np.round(np.exp(logit), decimals=5)
                else:
                    prob = 0
                final_attribute.append(prob)
        final_attributes.append(final_attribute)
    final_attributes = np.array(final_attributes)
    k = len(final_attributes[0])
    sum_attributes = final_attributes.sum(axis=1) / k
    baseline_attribute = np.array([list(item.values())[0] for item in baseline])
    baseline_attribute = sum(baseline_attribute) / len(baseline_attribute)
    importance_scores = (1 - sum_attributes)
    norm_scores = importance_scores / sum(importance_scores)
    print(encoded_prompt)
    assert (len(norm_scores) == len(encoded_prompt))
    final_attributes_dict = [{
        'token': strip_tokenizer_prefix(encoded_prompt[i]),
        'type': 'input',
        'value': norm_scores[i],
        'position': i
    } for i, item in enumerate(encoded_prompt)]
    print(f"\n baseline {final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def similarity_method(baseline, candidates, encoded_prompt):
    """
    Calculate the similarity between the baseline and candidates.
    """

    baseline_embedding = get_embedding(baseline)
    print(f"\nbaseline_embedding{baseline_embedding}")
    candidate_embeddings = [get_embedding(candidate) for candidate in candidates]
    similarities = []
    for candidate_embedding in candidate_embeddings:
        similarity = np.dot(baseline_embedding, candidate_embedding) / (
                    np.linalg.norm(baseline_embedding) * np.linalg.norm(candidate_embedding))
        similarities.append(1 - similarity)
    print(f"\nsimilarities{similarities}")
    norm_scores = similarities / sum(similarities)
    print(f"\nsimilarities{norm_scores}")

    final_attributes_dict = [{
        'token': strip_tokenizer_prefix(encoded_prompt[i]),
        'type': 'input',
        'value': norm_scores[i],
        'position': i
    } for i, item in enumerate(encoded_prompt)]
    print(f"\n baseline {final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }


def do_comparison(cleaned_baseline, candidate_token):
    comparison_set = set(cleaned_baseline)
    candidate_set = set(candidate_token)
    marks = [1 if token in candidate_set else 0 for token in comparison_set]
    average = sum(marks) / len(comparison_set)
    return average


def discretize_logits_method(baseline, candidates, encoded_prompt):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    baseline_tokens = encoding.encode(baseline)
    baseline_tokens = [encoding.decode_single_token_bytes(token) for token in baseline_tokens]
    # print(f"baseline_tokens{baseline_tokens}")
    cleaned_baseline = [strip_tokenizer_prefix(token) for token in baseline_tokens]
    scores = []
    for candidate in candidates:
        candidate_token = encoding.encode(candidate)
        candidate_token = [encoding.decode_single_token_bytes(token) for token in candidate_token]
        cleaned_candidate = [strip_tokenizer_prefix(token) for token in candidate_token]
        score = do_comparison(cleaned_baseline, cleaned_candidate)
        scores.append(1 - score)
    scores = np.array(scores)
    norm_scores = scores / np.sum(scores)
    final_attributes_dict = [{
        'token': strip_tokenizer_prefix(encoded_prompt[i]),
        'type': 'input',
        'value': norm_scores[i],
        'position': i
    } for i, item in enumerate(encoded_prompt)]
    print(f"\n baseline {final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }


def analyze_important_words(probabilities, encoded_prompt):
    """
    Analyze the probabilities of each token in the given prompt.
    """
    baseline = preprocess_baseline(probabilities[0])
    masked_words = preprocess_baseline(probabilities[1:])
    importance_score_tokens = calculate_importance_score_tokens(baseline, masked_words, encoded_prompt)

    return importance_score_tokens


def calculate(prompt, component_sentences):
    try:
        import time
        start_time = time.time()
        candidates, encoded_prompt, total_input = mask_each_tokens_for_openai(prompt)
        total_problogits, response, total_output,responses = inference_openai(candidates)
        end_time = time.time()
        tokens_importance = analyze_important_words(total_problogits, encoded_prompt)
        words_importance = calculate_word_scores(prompt, tokens_importance)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return tokens_importance, words_importance, component_importance, response, end_time - start_time, total_input, total_output,responses
    except:
        return None, None, None, None, None, None, None,None


def calculate_discretize(prompt, component_sentences):
    try:
        import time
        start_time = time.time()
        candidates, encoded_prompt, total_input = mask_each_tokens_for_openai(prompt)
        response, output_cost = infer_openai_without_logits(candidates)
        end_time = time.time()
        tokens_importance = discretize_logits_method(response[0], response[1:], encoded_prompt)
        words_importance = calculate_word_scores(prompt, tokens_importance)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return tokens_importance, words_importance, component_importance, response[
            0], end_time - start_time, total_input, output_cost,response
    except:
        return None, None, None, None, None, None, None,None


def calculate_similarity(prompt, component_sentences):
    try:
        import time
        start_time = time.time()
        candidates, encoded_prompt, total_input = mask_each_tokens_for_openai(prompt)
        response, output_cost = infer_openai_without_logits(candidates)
        end_time = time.time()
        tokens_importance = similarity_method(response[0], response[1:], encoded_prompt)
        words_importance = calculate_word_scores(prompt, tokens_importance)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return tokens_importance, words_importance, component_importance, response[
            0], end_time - start_time, total_input, output_cost,response
    except:
        return None, None, None, None, None, None, None,None


def run_initial_inference(start, end, method):
    df = load_and_preprocess([start, end])

    data = []
    for ind, example in enumerate(df.select(range(len(df)-1))):
        if method == "similarity":
            token, word, component, real_output, exec_time, total_input, output_cost,response = calculate_similarity(
                example['prefix_query'], example['component_range'])
        elif method == "discretize":
            token, word, component, real_output, exec_time, total_input, output_cost,response = calculate_discretize(
                example['prefix_query'], example['component_range'])
        elif method == "logits":
            token, word, component, real_output, exec_time, total_input, output_cost,response = calculate(
                example['prefix_query'], example['component_range'])
        print("id----------->", ind, exec_time, total_input, output_cost)
        if token is not None:
            data.append(
                {'prompt': example['prefix_query'], "real_output": real_output, "token_level": token, "word_level": word,
                 "component_level": component,
                 'instruction': example['instruction'],
                 'query': example['query'],
                 "component_range": example['component_range'],  # TODO: not a list, its a dict
                 "instruction_weight": component[0].get("instruction"),
                 "query_weight": component[0].get("query"),
                 "exec_time": exec_time,
                 "input_cost": total_input,
                 "output_cost": output_cost,
                 "total_cost": total_input + output_cost,
                 "real_ouputs":response

                 }
            )
        else:
            print(f"openai_infer.py:186   No output for prompt: {example['prefix_query']}")
    result = pd.DataFrame(data)

    return result


def only_calculate_results(prompt):
    _, response, output_cost = inference_openai([prompt])
    return response


def run_peturbed_inference(df, results_path, column_names=None):
    if column_names is None:
        # getting the columns demarkated by `reconstructed`
        column_names = []
        for col in df.columns:
            if '_reconstructed_' in col:
                if "0.2" in col:
                    column_names.append(col)

    print("running inference on petrubation columns:", column_names)

    for id, col_name in enumerate(column_names):
        df[col_name + "_result"] = df.apply(lambda row: only_calculate_results(row[col_name]), axis=1)
    # df.to_pickle("sentence" + str(id) +"_intermediate-run_peturbed_inference.pkl")
    print("sentence has done!")

    return df


def main(method,start,end):

    inference_df = run_initial_inference(start=start, end=end, method=method)
    inference_df.to_pickle(f"{start}_{end}_{method}_flesh_new_prompt_qa_inferenced_df.pkl")
    print("\ndone the inference")

    with open(f"{start}_{end}_{method}_flesh_new_prompt_qa_inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)
    postprocess_inferenced_df = postproces_inferenced(postprocess_inferenced_df)
    postprocess_inferenced_df.to_pickle(f"{start}_{end}_{method}_flesh_new_prompt_qa_postprocess_inferenced_df.pkl")
    print("\n done the postprocess")






if __name__ == "__main__":
    start = 1303
    end = start + 100
    main("similarity",1303, 1403)
    main("similarity", 1003, 1103)
    main("similarity", 1403, 1503)
    # main("discretize")
    # main("logits")




