from nltk.tokenize import word_tokenize
import pandas as pd
import torch

def generate_original_tokens_level_attribute(attribution, tokens):
    """
    Generate original tokens level attribute from the attribution.

    Parameters:
    - attribution: The attribution for the tokens.
    - tokens: The original tokens.

    Returns:
    - original_tokens_level_attribute: The attribute for the original tokens.
    """
    original_tokens_level_attribute = []
    for i, (token, attribution_value) in enumerate(zip(tokens, attribution)):
        each_level_attribute = {'token': token,
                                'type': 'input',
                                'value': attribution_value,
                                'position': id}
        original_tokens_level_attribute.append(each_level_attribute)
    return original_tokens_level_attribute
def calculate_word_scores( model_input, data):
    """
    Calculate scores for each word in the input sentence.

    Parameters:
    - model_input: A string containing the sentence to be processed.
    - data: A dictionary with a 'tokens' key, containing the contributions data.

    Returns:
    - A list of dictionaries with scored tokens.
    """
    words = word_tokenize(model_input)
    contributions = data.get('tokens')
    index = 0
    total_value = 0
    real_token = ''
    combined_contributions = []

    for id, word in enumerate(words, start=0):
        while index < len(contributions):
            token = contributions[index].get('token')
            total_value += float(contributions[index].get('value'))
            real_token += token
            index += 1

            if len(real_token) == len(word):
                combined_contributions.append({
                    'token': real_token,
                    'type': 'input',
                    'value': total_value,
                    'position': id
                })
                total_value = 0
                real_token = ''
                break

    return {'tokens': combined_contributions}


def preprocess_attributes_values(attributes):
    """

    :param attributes:
        attributions calculated by the perturbation or gradient method.

    :return:
        dictionary of attributes - token level
    """

    attr_res = attributes.token_attr.cpu().detach().numpy()
    absolute_attr_res = np.absolute(attr_res)
    norma_attr_res = np.sum(absolute_attr_res,axis=0)
    normalized_attr_res = norma_attr_res / np.sum(norma_attr_res)
    tokens_attr_dict = generate_original_tokens_level_attribute(normalized_attr_res)
    return tokens_attr_dict
