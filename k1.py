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
#from qa_preprocess import load_and_preprocess
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
    model.bfloat16()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"

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
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=50,temperature=0.01)[0]
        generated_tokens = output_ids[len(model_input["input_ids"][0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #print(f"Generated text: {response}")
    return response



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

    outputs = model.generate(**inputs, temperature=0.01, output_logits=True, max_new_tokens=50,
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

    outputs = model.generate(**inputs, temperature=0.01, output_logits=True, max_new_tokens=50,
                             return_dict_in_generate=True, output_scores=True)
    response = tokenizer.decode(outputs['sequences'][0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    #print(outputs)
    all_top_logits = []
    # print(outputs.scores)
    #print(outputs['sequences'][0][len(inputs["input_ids"][0]):])
    tensor_list = outputs['sequences'][0][len(inputs["input_ids"][0]):].tolist()
    for i,id in enumerate(tensor_list):
        log_probabilities = (outputs.logits)[i]
        top_logits= log_probabilities[0][id]
        all_top_logits.append((top_logits))
    top_indices = sorted(range(len(all_top_logits)), key=lambda x: all_top_logits[x], reverse=True)[:5]

    return response,top_indices

def perturbation_attribution_top_k(model, tokenizer, prompt):
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

    nested_list = [weight[i::2] for i in range(2)]
    np_array = np.array(nested_list)
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
    end_time = time.time()
    return {
        "tokens": final_attributes_dict
    },target,end_time - start_time, gpu_memory_usage

def do_comparison(cleaned_baseline, candidate_token):
    comparison_set = set(cleaned_baseline)
    candidate_set = set(candidate_token)
    marks = [1 if token in candidate_set else 0 for token in comparison_set]
    average = sum(marks) / len(comparison_set)
    return average
#     return mask
def generated_tensor_candidate(baseline):
    input_tensor = baseline.squeeze(0)
    #print(input_tensor)# Starting with a simple 1D tensor

    # Get the number of elements in the tensor
    result_tensors = [input_tensor[torch.arange(input_tensor.size(0)) != i] for i in range(input_tensor.size(0))]


    result_tensor = torch.stack(result_tensors)
    result_tensor = result_tensor.to(device=0)


    return result_tensor


def generate_candidate(original_prompt, tokenizer):
    baseline_input = tokenizer(original_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    candidate_input = generated_tensor_candidate(baseline_input["input_ids"])
    return candidate_input
def similarity_method(model, tokenizer, prompt):
    model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    real_length = len(model_input['input_ids'][0][:])
    candidate_input = generate_candidate(prompt, tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'].squeeze(0))
    import time
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(candidate_input, max_new_tokens=50, temperature=0.1)
        response = tokenizer.batch_decode(output_ids[:, real_length:], skip_special_tokens=True)
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=50, temperature=0.1)[0]
        baseline_input = tokenizer.decode(output_ids[len(model_input['input_ids'][0][:]):], skip_special_tokens=True)

    # Load the model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device=0)
    gpu_memory_usage = torch.cuda.max_memory_allocated(device=0)
    gpu_memory_usage = gpu_memory_usage/1024/1024/1204
    print(f"GPU Memory Usage: {gpu_memory_usage} GB")

    # Define the query and the list of sentences
    query = baseline_input
    sentences = response

    # Encode the query and sentences to get their embeddings
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = sentence_model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)

    # Find the most similar sentences
    similarities = cosine_scores[0].cpu().numpy()

    similarities = 1 - similarities

    similarities = similarities / sum(similarities)
    final_attributes_dict = [{
        'token': hg_strip_tokenizer_prefix(tokens[i]),
        'type': 'input',
        'value': similarities[i],
        'position': i
    } for i, item in enumerate(tokens)]
    print(f"\n baseline {final_attributes_dict}")
    print("prompt",prompt)
    print(similarities)
    end_time = time.time()
    return {
        "tokens": final_attributes_dict
    }, baseline_input, end_time - start_time, gpu_memory_usage



def discretize_method(model, tokenizer, prompt):
    model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    real_length = len(model_input['input_ids'][0][:])
    candidate_input = generate_candidate(prompt, tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'].squeeze(0))
    import time
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(candidate_input, max_new_tokens=50, temperature=0.1)
        #     print(output_ids)
        response = tokenizer.batch_decode(output_ids[:, real_length:], skip_special_tokens=True)
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=50, temperature=0.1)[0]
        baseline_input = tokenizer.decode(output_ids[len(model_input['input_ids'][0][:]):], skip_special_tokens=True)


    gpu_memory_usage = torch.cuda.max_memory_allocated(device=0)
    gpu_memory_usage = gpu_memory_usage/1024/1024/1204
    print(f"GPU Memory Usage: {gpu_memory_usage} GB")
    scores = []
    baseline_input_tokens = tokenizer.tokenize(baseline_input)
    tokenized_texts_tokens = [tokenizer.tokenize(text) for text in response]
    print(baseline_input_tokens)
    print(tokenized_texts_tokens)
    for token in tokenized_texts_tokens:
        score = do_comparison(baseline_input_tokens,token)
        scores.append(1 - score)
    scores = np.array(scores)
    norm_scores = scores / np.sum(scores)
    print(len(norm_scores))
    print(len(tokens))
    end_time  = time.time()
    final_attributes_dict = [{
        'token': hg_strip_tokenizer_prefix(item),
        'type': 'input',
        'value': norm_scores[i],
        'position': i
    } for i, item in enumerate(tokens)]
    print(f"\n baseline {final_attributes_dict}")
    print("prompt",prompt)
    print(norm_scores)
    return {
        "tokens": final_attributes_dict
    }, baseline_input, end_time - start_time, gpu_memory_usage


def perturbation_attribution(model, tokenizer, prompt):
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
def process_logits(result, baseline_output_ids,bb):
    print(baseline_output_ids)
    baseline_final_socre = []
    a = 0
    for ind, each_logit in enumerate(result.logits):
        log_probs = torch.nn.functional.log_softmax(each_logit, dim=1)

        baseline_final_socre.append(log_probs[0][baseline_output_ids[ind]])
        if a == bb-1:
            break
        a+=1
    values_list = [x.item() for x in baseline_final_socre]
    values_list = np.array(values_list)
    exp_array = np.exp(values_list)
    # print(values_list)
    return exp_array.reshape(1, -1)


def process_logits_candidate(result, baseline_output_ids,bb):
    # print(baseline_output_ids)
    baseline_final_socre = []
    a = 0
    for ind, each_logit in enumerate(result.logits):
        log_probs = torch.nn.functional.log_softmax(each_logit, dim=1)
        baseline_final_socre.append(log_probs[:, baseline_output_ids[ind]].tolist())
        if a == bb-1:
            break
        a+=1

    import numpy as np
    baseline_final_socre = np.array(baseline_final_socre)
    baseline_final_socre = np.exp(baseline_final_socre)
    transposed_array = baseline_final_socre.T
    # print(baseline_final_socre)
    return transposed_array
def new_logit_parallel(model, tokenizer, prompt, max_new_tokens):
    model_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    real_length = len(model_input['input_ids'][0][:])
    candidate_input = generate_candidate(prompt, tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'].squeeze(0))
    import time
    start_time = time.time()
    tenseor_List = []
    with torch.no_grad():
        result = model.generate(model_input["input_ids"], temperature=0.1, max_new_tokens=50,
                                return_dict_in_generate=True, output_scores=True, output_logits=True)
        baseline_output_ids = result[0]
        print('baseline_output_ids',baseline_output_ids[0][real_length:])
        a = len(baseline_output_ids[0][real_length:])
        gpu_memory_usage = torch.cuda.max_memory_allocated(device=0)
        gpu_memory_usage = gpu_memory_usage / 1024 / 1024 / 1204
        print(f"GPU Memory Usage: {gpu_memory_usage} GB")

        for i, batch in enumerate(candidate_input):

            candidate_result = model.generate(batch, temperature=0.1, output_logits=True,
                                          max_new_tokens=50,
                                          return_dict_in_generate=True, output_scores=True)

            candidate_result_respone = candidate_result[0]

            b = candidate_result_respone[:, real_length - 1:].size()[1]
            if a >= b:
                bb = b
            else:
                bb = a

            baseline_logits = process_logits(result, baseline_output_ids[0][real_length:],bb)
            candidate_logits = process_logits_candidate(candidate_result, baseline_output_ids[0][real_length:],bb)
            tenseor_List.append(candidate_logits)

        baseline_input = tokenizer.decode(baseline_output_ids[0][real_length:],
                                          skip_special_tokens=True)

        min_columns = min(array.shape[1] for array in tenseor_List)

        adjusted_arrays = [array[:, :min_columns] for array in tenseor_List]

        concatenated_array = np.concatenate(adjusted_arrays, axis=0)
        min_columns = min(concatenated_array.shape[1], baseline_logits.shape[1])
        adjusted_concatenated_array = concatenated_array[:, :min_columns]
        adjusted_another_array = baseline_logits[:, :min_columns]
        attribute = adjusted_another_array - adjusted_concatenated_array
        real_attr_res = np.absolute(attribute)
        real_attr_res = np.sum(real_attr_res, axis=1)

        newer_sum_normalized_array = real_attr_res / np.sum(real_attr_res)
        #print(newer_sum_normalized_array)

        final_attributes_dict = [{
            'token': hg_strip_tokenizer_prefix(tokens[i]),
            'type': 'input',
            'value': newer_sum_normalized_array[i],
            'position': i
        } for i, item in enumerate(tokens)]
        end_time = time.time()

        return {
            "tokens": final_attributes_dict
        }, baseline_input, end_time - start_time, gpu_memory_usage



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

    step_list = top_indices
    #print(step_list)
    attr_res = llm_attr.attribute(inp=inp,target= response,step_list=step_list, n_steps=10)
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

def gradient_attribution(model, tokenizer, prompt):
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
    attr_res = llm_attr.attribute(inp, target=target,n_steps=2)
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
    print(f"{final_attributes_dict}")
    return {
        "tokens": final_attributes_dict
    }, target, end_time - start_time, gpu_memory_usage




def calculate_attributes(prompt,component_sentences,model,tokenizer,method):
    """
    Calculate the attributions for the given model and calculate_method.

    Parameters:
    - model_weight: If True, use the model weights as input for the attribution calculation.
    - calculate_method: The method to calculate the attribution (e.g., perturbation, SHAP, etc.).

    Returns:
    - attribution: The attributions calculated using the given calculate_method.
    """
    calculate_method = method
    model_weight = False
    if calculate_method == "perturbation":
        attribution,target,time,gpu_memory_usage = perturbation_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage
    elif calculate_method == "new_gradient":
        attribution, target, time, gpu_memory_usage = new_gradient_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance,component_importance, target, time, gpu_memory_usage
    elif calculate_method == "new_perturbation":
        attribution, target, time, gpu_memory_usage = discretize_method(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target, time, gpu_memory_usage

    elif calculate_method == "gradient":
        attribution,target,time,gpu_memory_usage = gradient_attribution(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage
    elif calculate_method == "top_k_perturbation":
        attribution,target,time,gpu_memory_usage = perturbation_attribution_top_k(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage
    elif calculate_method == "similarity":
        attribution,target,time,gpu_memory_usage = similarity_method(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage
    else:
        attribution,target,time,gpu_memory_usage = discretize_method(model, tokenizer, prompt=prompt)
        words_importance = calculate_word_scores(prompt, attribution)
        component_importance = calculate_component_scores(words_importance.get('tokens'), component_sentences)
        return attribution, words_importance, component_importance, target,time,gpu_memory_usage



    # if model_weight:
    #     pass
    # else:
    #     pass
def run_initial_inference(start, end,model,tokenizer,method,df):

    print(len(df))
    data = []

    for ind, example in enumerate(df.select(range(len(df)-1))):

            token, word, component, real_output,exec_time,gpu_memory_usage = calculate_attributes(example['sentence'], example['component_range'],model,tokenizer,method)
            print("token--------",token)
            print(word)
            print(component)
            if token is not None:
                data.append(
                    {'prompt': example['sentence'], "real_output": real_output, "token_level": token, "word_level": word,
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




def main(method,model, tokenizer,df,start,end ):

   # method = "gradient"
    inference_df = run_initial_inference(start=start,end=end,model=model,tokenizer=tokenizer,method=method,df=df)
    inference_df.to_pickle(f"{start}_{end}_{method}_llama2_sa_new_inferenced_df.pkl")
    print("\ndone the inference")

    # with open(f"{start}_{end}_{method}_1024_llama2_qa_new_inferenced_df.pkl", "rb") as f:
    #     postprocess_inferenced_df = pickle.load(f)
    # postprocess_inferenced_df = postproces_inferenced(postprocess_inferenced_df)
    # postprocess_inferenced_df.to_pickle(f"{start}_{end}_{method}_1024_llama2_qa_new_postprocess_inferenced_df.pkl")
    # print("\n done the postprocess")




if __name__ == "__main__":
    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf", BitsAndBytesConfig(bits=4, quantization_type="fp16"))
    start = 16000
    end = start + 10
    df = load_and_preprocess([start, end])

    #main("gradient")
    # print(df[16000])
    # print(df[24000])
    # print(df[32000])
    # print(df[40000])
    # print(df[56000])

    #main("new_gradient",model, tokenizer,df,start, end )
    main("similarity",model, tokenizer,df,start, end )
    main("discretize",model, tokenizer,df,start, end )
    main("new_perturbation", model, tokenizer, df, start, end)
    start = 24000
    end = start + 10
    df = load_and_preprocess([start, end])

    main("similarity",model, tokenizer,df,start, end )
    main("discretize",model, tokenizer,df,start, end )
    main("new_perturbation", model, tokenizer, df, start, end)

    start = 32000
    end = start + 10
    df = load_and_preprocess([start, end])

    main("similarity", model, tokenizer, df, start, end)
    main("discretize", model, tokenizer, df, start, end)
    main("new_perturbation", model, tokenizer, df, start, end)

    start = 56000
    end = start + 10
    df = load_and_preprocess([start, end])

    main("similarity", model, tokenizer, df, start, end)
    main("discretize", model, tokenizer, df, start, end)
    main("new_perturbation", model, tokenizer, df, start, end)






