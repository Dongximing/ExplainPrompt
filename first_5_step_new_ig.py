import bitsandbytes as bnb
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys
import numpy as np
import pandas as pd
from utils import calculate_word_scores, calculate_component_scores, hg_strip_tokenizer_prefix, postproces_inferenced
from qa_preprocess import load_and_preprocess

from captum.attr import (
    FeatureAblation,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
)
import pickle



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
        model_name,torch_dtype=torch.float16, device_map="auto"
    )
    model.bfloat16()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def generate_text_with_ig(model, tokenizer, current_input, max_new_tokens,bl=False):
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

    outputs = model.generate(**inputs, temperature=0.01, output_logits=True, max_new_tokens=100,
                             return_dict_in_generate=True, output_scores=True)
    response = tokenizer.decode(outputs['sequences'][0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    all_top_logits = []

    tensor_list = outputs['sequences'][0][len(inputs["input_ids"][0]):].tolist()
    for i,id in enumerate(tensor_list):
        log_probabilities = (outputs.logits)[i]
        top_logits= log_probabilities[0][id]
        all_top_logits.append((top_logits))
    top_indices = sorted(range(len(all_top_logits)), key=lambda x: all_top_logits[x], reverse=True)[:5]

    return response,top_indices






def new_gradient_attribution(model, tokenizer, prompt):
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
    response, top_indices = generate_text_with_ig(model, tokenizer, prompt,100)
    emb_layer = model.get_submodule("model.embed_tokens")
    ig = LayerIntegratedGradients(model, emb_layer)
    llm_attr = LLMGradientAttribution(ig, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )

    step_list = [0,1,2,3,4]
    #print(step_list)
    attr_res = llm_attr.attribute(inp=inp,target=response,step_list=step_list, n_steps=10)
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
    print(f"response---------》{response}")
    return {
        "tokens": final_attributes_dict
    }, response, end_time - start_time, gpu_memory_usage



def calculate_attributes(prompt,component_sentences,model,tokenizer,method):
    """
    Calculate the attributions for the given model and calculate_method.

    Parameters:
    - model_weight: If True, use the model weights as input for the attribution calculation.
    - calculate_method: The method to calculate the attribution (e.g., perturbation, SHAP, etc.).

    Returns:
    - attribution: The attributions calculated using the given calculate_method.
    """

    attribution, target, time, gpu_memory_usage = new_gradient_attribution(model, tokenizer, prompt=prompt)
    words_importance = calculate_word_scores(prompt, attribution)
    component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
    return attribution, words_importance, component_importance, target, time, gpu_memory_usage



def run_initial_inference(start, end,model,tokenizer,method,df):

    print(len(df))
    data = []

    for ind, example in enumerate(df.select(range(len(df)-1))):

            token, word, component, real_output,exec_time,gpu_memory_usage = calculate_attributes(example['prefix_query'], example['component_range'],model,tokenizer,method)

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
                     "gpu_memory_usage":gpu_memory_usage
                     }
                )
            else:
                print(f"hg_infer.py:170  No output for prompt: {example['sentence']}")
    result = pd.DataFrame(data)

    return result





def main(method,model, tokenizer,df,start,end ):

   # method = "gradient"
    inference_df = run_initial_inference(start=start,end=end,model=model,tokenizer=tokenizer,method=method,df=df)
    inference_df.to_pickle(f"{start}_{end}_{method}_qa_new_inferenced_df.pkl")
    print("\ndone the inference")

    with open(f"{start}_{end}_{method}_qa_new_inferenced_df.pkl", "rb") as f:
        postprocess_inferenced_df = pickle.load(f)
    postprocess_inferenced_df = postproces_inferenced(postprocess_inferenced_df)
    postprocess_inferenced_df.to_pickle(f"{start}_{end}_{method}_qa_new_postprocess_inferenced_df.pkl")
    print("\n done the postprocess")




if __name__ == "__main__":
    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf", BitsAndBytesConfig(bits=4, quantization_type="fp16"))
    start = 5303
    end = start + 200
    df = load_and_preprocess([start, end])

    main("new_gradient",model, tokenizer,df,start, end )




