import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys
import numpy as np
from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)
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
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=5)[0]
        generated_tokens = output_ids[len(model_input["input_ids"][0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response





def perturbation_attribution(model, tokenizer,prompt,**kwargs):
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

    return attr_res


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
    attribution = LayerIntegratedGradients(model)
    llm_attr = LLMGradientAttribution(attribution, tokenizer)
    inp = TextTokenInput(
        prompt,
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp)

    return attr_res

def openai_logit_attribution(model):



def calculate_attributes(model_weight=False, calculate_method="perturbation"):
    """
    Calculate the attributions for the given model and calculate_method.

    Parameters:
    - model_weight: If True, use the model weights as input for the attribution calculation.
    - calculate_method: The method to calculate the attribution (e.g., perturbation, SHAP, etc.).

    Returns:
    - attribution: The attributions calculated using the given calculate_method.
    """
    if calculate_method == "perturbation":
        attribution = perturbation_attribution(self.model, self.tokenizer)
    elif calculate_method == "gradient":
        attribution = (self.model, self.tokenizer)
        if model_weight:
            attribution.attribute(self.model.get_input_embeddings().weight, target=self.target_token_id)
        else:
            attribution.attribute(self.input_tokens, target=self.target_token_id)

    elif calculate_method == "llm_attribution":
        attribution = LLMAttribution(self.model,
