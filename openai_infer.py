import pandas as pd
from openai import OpenAI
from math import exp
import numpy as np
# from IPython.display import display, HTML
import os
import tiktoken
from datasets import load_dataset
from utils import calculate_word_scores, calculate_component_scores, strip_tokenizer_prefix, postproces_inferenced
from data_read_preprocess import load_and_preprocess
from peturbation import run_peturbation, do_peturbed_reconstruct
import pickle

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                                       ""))

CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline: {headline}"""


def get_completion(
        messages: list[dict[str, str]],
        model: str = "gpt-turbo-3.5",
        max_tokens=500,
        temperature=0,
        stop=None,
        seed=123,
        tools=None,
        logprobs=None,
        # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
        top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def inference_openai(sentences):
    list_top20_logprobs = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            top_logprobs = 1
        else:
            top_logprobs = 20  # if we want to get top 20 logprobs, otherwise 20
        API_RESPONSE = get_completion(
            [{"role": "user", "content": sentence}],
            model="gpt-3.5-turbo-0125",
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        top_twenty_logprobs = API_RESPONSE.choices[0].logprobs
        # print(f"\n top_twenty_logprobs{top_twenty_logprobs}")
        response = API_RESPONSE.choices[0].message.content
        # print(f"\n sentence {sentence}")
        # print(f"\n response {response}")
        token_dict = {
            data.token: {tp.token: tp.logprob for tp in data.top_logprobs}
            for data in top_twenty_logprobs.content
        }
        # print(f"\n token_dict {token_dict}")
        list_top20_logprobs.append(token_dict)

    return list_top20_logprobs, response
    # print(f"\ntop_twenty_logprobs: {top_twenty_logprobs}")


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
    for masked_list in masked_lists:
        masked_sentence = encoding.decode(masked_list)
        masked_sentences.append(masked_sentence)
    masked_sentences.insert(0, prompt)
    encoded_prompt = [encoding.decode_single_token_bytes(token) for token in tokens]
    return masked_sentences, encoded_prompt


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
    baseline = [{key: np.round(np.exp(value), decimals=5) for key, value in item.items()} for item in baseline]
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
    # print(f"final_attributes{final_attributes}")
    k = len(final_attributes[0])
    sum_attributes = final_attributes.sum(axis=1) / k
    baseline_attribute = np.array([list(item.values())[0] for item in baseline])
    baseline_attribute = sum(baseline_attribute) / len(baseline_attribute)
    #print(f"\nbaseline{baseline_attribute}")
    importance_scores = (1 - sum_attributes)
    # print(f"\nimportance_scores{importance_scores}")
    norm_scores = importance_scores / sum(importance_scores)
    # print(f"\nfinal_attributes  {baseline}")
    # print(f"\nfinal_    {encoded_prompt}")
    assert (len(norm_scores) == len(encoded_prompt))
    final_attributes_dict = [{
        'token': strip_tokenizer_prefix(encoded_prompt[i]),
        'type': 'input',
        'value': norm_scores[i],
        'position': i
    } for i, item in enumerate(encoded_prompt)]
    # print(f"\n baseline {final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }


def analyze_important_words(probabilities, encoded_prompt):
    """
    Analyze the probabilities of each token in the given prompt.
    """
    baseline = preprocess_baseline(probabilities[0])
    masked_words = preprocess_baseline(probabilities[1:])

    # print(f"baseline {masked_words}")

    importance_score_tokens = calculate_importance_score_tokens(baseline, masked_words, encoded_prompt)

    return importance_score_tokens


def calculate(prompt, component_sentences):
    try:
        print("prompt",prompt)
        candidates, encoded_prompt = mask_each_tokens_for_openai(prompt)
        total_problogits, response = inference_openai(candidates)
        tokens_importance = analyze_important_words(total_problogits, encoded_prompt)
        words_importance = calculate_word_scores(prompt, tokens_importance)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return tokens_importance, words_importance, component_importance, response
    except:
        return None, None, None, None


def run_initial_inference(start, end):
    df = load_and_preprocess([start,end])
    print(len(df))
    data = []
    for ind, example in enumerate(df.select(range(len(df)-1))):

            token, word, component, real_output = calculate(example['sentence'], example['component_range'])
            print("component----------->",component[2])
            if token is not None:
                data.append(
                    {'prompt': example['sentence'], "real_output": real_output, "token_level": token, "word_level": word,
                     "label": example['label'],
                     "component_level": component,
                     'instruction': example['instruction'],
                     'query': example['query'],
                     "component_range": example['component_range'],  # TODO: not a list, its a dict
                     "instruction_weight": component[0].get("instruction"),
                     "query_weight": component[0].get("query")

                     }
                )
            else:
                print(f"openai_infer.py:186   No output for prompt: {example['sentence']}")
    result = pd.DataFrame(data)

    return result
def only_calculate_results(prompt):
    
    _, response = inference_openai([prompt])
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
        df[col_name+"_result"] = df.apply(lambda row: only_calculate_results(row[col_name]), axis=1)
    # df.to_pickle("sentence" + str(id) +"_intermediate-run_peturbed_inference.pkl")
    print("sentence has done!")


    return df


if __name__ == "__main__":
    start = 45650
    end = start + 350
    inference_df = run_initial_inference(start=start,end=end)
    inference_df.to_pickle(f"{start}_{end}inferenced_df.pkl")
    print("\ndone the inference")

    with open(f"{start}_{end}inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)
    postprocess_inferenced_df = postproces_inferenced(postprocess_inferenced_df)
    postprocess_inferenced_df.to_pickle(f"{start}_{end}postprocess_inferenced_df.pkl")
    print("\n done the postprocess")


    with open(f"{start}_{end}postprocess_inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)

    perturbed_df = run_peturbation(postprocess_inferenced_df.copy())
    perturbed_df.to_pickle(f"{start}_{end}perturbed_df.pkl")
    print("\n done the perturbed")

    with open(f"{start}_{end}perturbed_df.pkl", "rb") as f:
        reconstructed_df = pickle.load(f)
    reconstructed_df = do_peturbed_reconstruct(reconstructed_df.copy(), None)
    reconstructed_df.to_pickle(f"{start}_{end}reconstructed_df.pkl")
    print("\n done the reconstructed")

    with open(f"{start}_{end}reconstructed_df.pkl", "rb") as f:
        reconstructed_df = pickle.load(f)
    perturbed_inferenced_df = run_peturbed_inference(reconstructed_df, results_path=None, column_names=None)
    perturbed_inferenced_df.to_pickle(f"{start}_{end}perturbed_inferenced_df.pkl")
    print("\n done the reconstructed inference data")







