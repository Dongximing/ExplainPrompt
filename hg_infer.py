import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys
import numpy as np
import pandas as pd
from utils import calculate_word_scores, calculate_component_scores, hg_strip_tokenizer_prefix, postproces_inferenced
from data_read_preprocess import load_and_preprocess
from peturbation import run_peturbation, do_peturbed_reconstruct
from captum.attr import (
    FeatureAblation,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
)
import pickle
from utils import generate_original_tokens_level_attribute


def load_model(model_name, bnb_config):
    """
    loading huggingface model.

    Parameters:
    - model_name: model name to load from the model directory (such as 'llama 3.1')
    - data: A dictionary with a 'tokens' key, containing the contributions data.

    Returns:
    - model: model and tokenizers for the model instance.
    """
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_text(model, tokenizer,input):
    """
    Generate text using the given model and tokenizer.

    Parameters:
    - model: The model to generate text.
    - tokenizer: The tokenizer used to tokenize the input.
    - input: The input string(prompt string)

    Returns:
        - response: The generated text based on the input prompt.
    """
    model_input = tokenizer(input, return_tensors="pt", padding=True, truncation=True).to("cuda")
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=2,temperature=0.01)[0]
        generated_tokens = output_ids[len(model_input["input_ids"][0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def generated_tensor_candidate(baseline):
    number_line = baseline.shape[1]
    output_tensor = baseline.repeat(number_line, 1)
    mask = torch.eye(number_line, dtype=bool)
    output_tensor[mask] = 2

    return output_tensor


def generate_candidate(original_prompt, tokenizer):
    baseline_input = tokenizer(original_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    candidate_input = generated_tensor_candidate(baseline_input["input_ids"])
    return candidate_input

def generate_text_with_logit(model, tokenizer, current_input, bl=True):
    """
    Generate text using the given model and tokenizer.

    Parameters:
    - model: The model to generate text.
    - tokenizer: The tokenizer used to tokenize the input.
    - input: The input string(prompt string)

    Returns:
        - response: The generated text based on the input prompt.
    """
    if type(current_input) == str:
        inputs = tokenizer([current_input], return_tensors="pt",add_special_tokens=False).to("cuda")
    else:
        inputs = tokenizer.decode(current_input[0])

        inputs = tokenizer([inputs], return_tensors="pt",add_special_tokens=False).to("cuda")

    outputs = model.generate(**inputs, temperature=0.01, output_logits=True, max_new_tokens=2,
                             return_dict_in_generate=True, output_scores=True)
    response = tokenizer.decode(outputs['sequences'][0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    # print(outputs)
    all_top_logits = []
    # print(outputs.scores)
    if bl:
        k = 20
    else:
        k = 1
    for i in range(len(outputs.scores)):
        log_probabilities = (outputs.logits)[i]
        top_logits, top_indices = torch.topk(log_probabilities, k)
        all_top_logits.append((top_indices[0], top_logits[0]))

    baselines = []
    for step, (indices, logits) in enumerate(all_top_logits):

        baseline = []
        for idx, logit in zip(indices, logits):
            token = tokenizer.decode([idx.item()])
            baseline.append(token)
        baselines.append(baseline)
    return baselines



def perturbation_attribution_top_k(model, tokenizer, prompt,**kwargs):
    """
    Calculate attribution using perturbation method.

    Parameters:
    - model: The model to calculate the attribution for.
    - tokenizer: The tokenizer used to tokenize the input.
    - prompt: The input tokens for which to calculate the attribution.
    - kwargs: Additional keyword arguments for the perturbation method.

    Returns:
    - attribution: The attributions calculated using the perturbation method.
    """

    target = generate_text(model, tokenizer, prompt)
    attribution = FeatureAblation(model)
    llm_attr = LLMAttribution(attribution, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp, target=target)
    real_attr_res = attr_res.token_attr.cpu().detach().numpy()
    real_attr_res = np.absolute(real_attr_res)
    baseline = generate_text_with_logit(model, tokenizer, prompt, bl=False)

    candidate_input = generate_candidate(prompt, tokenizer)
    weight = []
    for idx in range(candidate_input.shape[0]):  # Loop through each token column-wise
        # Select the column of tokens, ensuring it's in the correct shape ([batch_size, num_tokens])
        current_input = candidate_input[idx, :].unsqueeze(0)
        #print(current_input)
        # Generate or pass to the model
        with torch.no_grad():

            outputs = generate_text_with_logit(model, tokenizer, current_input, bl=True)
            for id, i in enumerate(baseline):
                bs = i[0]
                if bs in outputs[id]:
                    weight.append(1)
                else:
                    weight.append(0)
    print(len(weight))
    nested_list = [weight[i::2] for i in range(2)]
    np_array = np.array(nested_list)
    print(np_array)
    print(real_attr_res)
    real_attr_res = real_attr_res[:-1,:] * np_array



    real_attr_res = np.sum(real_attr_res,axis=0)

    labels = attr_res.input_tokens

    newer_sum_normalized_array = real_attr_res / np.sum(real_attr_res)
    final_attributes_dict = [{
        'token': hg_strip_tokenizer_prefix(labels[i]),
        'type': 'input',
        'value': newer_sum_normalized_array[i],
        'position': i
    } for i, item in enumerate(labels)]
    print(f"{final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    },target


def perturbation_attribution(model, tokenizer, prompt,**kwargs):
    """
    Calculate attribution using perturbation method.

    Parameters:
    - model: The model to calculate the attribution for.
    - tokenizer: The tokenizer used to tokenize the input.
    - prompt: The input tokens for which to calculate the attribution.
    - kwargs: Additional keyword arguments for the perturbation method.

    Returns:
    - attribution: The attributions calculated using the perturbation method.
    """
    import time
    start_time = time.time()
    target = generate_text(model, tokenizer, prompt)
    attribution = FeatureAblation(model)
    llm_attr = LLMAttribution(attribution, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp, target=target)
    gpu_memory_usage = torch.cuda.max_memory_allocated(device=0)
    gpu_memory_usage = gpu_memory_usage/1024/1024/1204
    print(f"GPU Memory Usage: {gpu_memory_usage} GB")
    real_attr_res = attr_res.token_attr.cpu().detach().numpy()
    real_attr_res = np.absolute(real_attr_res)
    real_attr_res = np.sum(real_attr_res,axis=0)
    labels = attr_res.input_tokens
    newer_sum_normalized_array = real_attr_res / np.sum(real_attr_res)
    final_attributes_dict = [{
        'token': hg_strip_tokenizer_prefix(labels[i]),
        'type': 'input',
        'value': newer_sum_normalized_array[i],
        'position': i
    } for i, item in enumerate(labels)]
    end_time = time.time()
    return {
        "tokens": final_attributes_dict
    },target, end_time - start_time, gpu_memory_usage





def gradient_attribution(model, tokenizer, prompt, kwargs):
    """
    Calculate attribution using gradient method.

    Parameters:
    - model: The model to calculate the attribution for.
    - tokenizer: The tokenizer used to tokenize the input.
    - prompt: The input tokens for which to calculate the attribution.
    - kwargs: Additional keyword arguments for the gradient method.

    Returns:
    - attribution: The attributions calculated using the gradient method.
    """
    import time
    start_time = time.time()
    target = generate_text(model, tokenizer, prompt)
    emb_layer = model.get_submodule("model.embed_tokens")
    ig = LayerIntegratedGradients(model, emb_layer)
    llm_attr = LLMGradientAttribution(ig, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp, target=target)
    gpu_memory_usage = torch.cuda.max_memory_allocated(device=0)
    real_attr_res = attr_res.token_attr.cpu().detach().numpy()
    real_attr_res = np.absolute(real_attr_res)
    real_attr_res = np.sum(real_attr_res, axis=0)
    labels = attr_res.input_tokens
    newer_sum_normalized_array = real_attr_res / np.sum(real_attr_res)
    final_attributes_dict = [{
        'token': hg_strip_tokenizer_prefix(labels[i]),
        'type': 'input',
        'value': newer_sum_normalized_array[i],
        'position': i
    } for i, item in enumerate(labels)]
    gpu_memory_usage = gpu_memory_usage/1024/1024/1204
    print(f"GPU Memory Usage: {gpu_memory_usage} GB")
    end_time = time.time()
    #print(f"{final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }, target, end_time - start_time, gpu_memory_usage




def calculate_attributes(prompt,component_sentences,model,tokenizer):
    """
    Calculate the attributions for the given model and calculate_method.

    Parameters:
    - model_weight: If True, use the model weights as input for the attribution calculation.
    - calculate_method: The method to calculate the attribution (e.g., perturbation, SHAP, etc.).

    Returns:
    - attribution: The attributions calculated using the given calculate_method.
    """
    calculate_method = "gradient"
    model_weight = False
    if calculate_method == "perturbation":
        attribution,target,time,gpu_memory_usage = perturbation_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage

    elif calculate_method == "gradient":
        attribution,target,time,gpu_memory_usage = gradient_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage
    if model_weight:
        pass
    else:
        pass
def run_initial_inference(start, end,model,tokenizer):
    df = load_and_preprocess([start,end])
    print(len(df))
    data = []

    for ind, example in enumerate(df.select(range(len(df)-1))):

            token, word, component, real_output,exec_time,gpu_memory_usage = calculate_attributes(example['sentence'], example['component_range'],model,tokenizer)
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
                     "query_weight": component[0].get("query"),
                     "exec_time": exec_time,
                     "gpu_memory_usage":gpu_memory_usage
                     }
                )
            else:
                print(f"hg_infer.py:170  No output for prompt: {example['sentence']}")
    result = pd.DataFrame(data)

    return result


def only_calculate_results(prompt,model, tokenizer):
    response = generate_text(model, tokenizer,prompt)
    return response


def run_peturbed_inference(df, model, tokenizer):
    column_names = None
    if column_names is None:
        # getting the columns demarkated by `reconstructed`
        column_names = []
        for col in df.columns:
              if '_reconstructed_' in col:
                if "0.2" in col:
                    column_names.append(col)

    print("running inference on petrubation columns:", column_names)

    for id, col_name in enumerate(column_names):
        df[col_name + "_result"] = df.apply(lambda row: only_calculate_results(row[col_name], model, tokenizer), axis=1)
    # df.to_pickle("sentence" + str(id) +"_intermediate-run_peturbed_inference.pkl")
    print("sentence has done!")
    return df


if __name__ == "__main__":
    start = 45000
    end = start +5

    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf", BitsAndBytesConfig(bits=4, quantization_type="fp16"))
    method = "gradient"
    inference_df = run_initial_inference(start=start,end=end,model=model,tokenizer=tokenizer)
    inference_df.to_pickle(f"{start}_{end}_{method}_new_inferenced_df.pkl")
    print("\ndone the inference")

    with open(f"{start}_{end}_{method}_new_inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)
    postprocess_inferenced_df = postproces_inferenced(postprocess_inferenced_df)
    postprocess_inferenced_df.to_pickle(f"{start}_{end}_{method}_new_postprocess_inferenced_df.pkl")
    print("\n done the postprocess")


    with open(f"{start}_{end}_{method}_new_postprocess_inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)

    perturbed_df = run_peturbation(postprocess_inferenced_df.copy())
    perturbed_df.to_pickle(f"{start}_{end}_{method}_new_perturbed_df.pkl")
    print("\n done the perturbed")

    with open(f"{start}_{end}_{method}_new_perturbed_df.pkl", "rb") as f:
        reconstructed_df = pickle.load(f)
    reconstructed_df = do_peturbed_reconstruct(reconstructed_df.copy(), None)
    reconstructed_df.to_pickle(f"{start}_{end}_{method}_new_reconstructed_df.pkl")
    print("\n done the reconstructed")

    with open(f"{start}_{end}_{method}_new_reconstructed_df.pkl", "rb") as f:
        reconstructed_df = pickle.load(f)
    perturbed_inferenced_df = run_peturbed_inference(reconstructed_df, model, tokenizer)
    perturbed_inferenced_df.to_pickle(f"{start}_{end}_{method}_new_perturbed_inferenced_df.pkl")
    print("\n done the reconstructed inference data")

