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
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=2)[0]
        generated_tokens = output_ids[len(model_input["input_ids"][0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response





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
    emb_layer = model.get_submodule("model.embed_tokens")
    ig = LayerIntegratedGradients(model, emb_layer)
    llm_attr = LLMGradientAttribution(ig, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp)

    return attr_res




def calculate_attributes(prompt,component_sentences,model,tokenizer):
    """
    Calculate the attributions for the given model and calculate_method.

    Parameters:
    - model_weight: If True, use the model weights as input for the attribution calculation.
    - calculate_method: The method to calculate the attribution (e.g., perturbation, SHAP, etc.).

    Returns:
    - attribution: The attributions calculated using the given calculate_method.
    """
    calculate_method = "perturbation"
    model_weight  = False
    if calculate_method == "perturbation":
        attribution,target = perturbation_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target

    elif calculate_method == "gradient":
        attribution = gradient_attribution(model, tokenizer, prompt=prompt)
    if model_weight:
        pass
    else:
        pass
def run_initial_inference(start, end,model,tokenizer):
    df = load_and_preprocess([start,end])
    print(len(df))
    data = []

    for ind, example in enumerate(df.select(range(len(df)-1))):

            token, word, component, real_output = calculate_attributes(example['sentence'], example['component_range'],model,tokenizer)
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
                print(f"hg_infer.py:170  No output for prompt: {example['sentence']}")
    result = pd.DataFrame(data)

    return result


def only_calculate_results(prompt):
    response = generate_text(model, tokenizer,prompt)
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


if __name__ == "__main__":
    start = 45070
    end = start + 3

    model, tokenizer = load_model("meta-llama/Llama-2-13b-chat-hf", BitsAndBytesConfig(bits=4, quantization_type="fp16"))

    inference_df = run_initial_inference(start=start,end=end,model= model,tokenizer = tokenizer)
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

