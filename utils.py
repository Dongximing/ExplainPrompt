from nltk.tokenize import word_tokenize
import pandas as pd
import re
import numpy as np
import random
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from sentence_transformers import SentenceTransformer, util
import copy
import nltk
nltk.download('words')
english_words = set(words.words())
# import torch

def strip_tokenizer_prefix(token):
    token = token.decode('utf-8')
    token = token.lstrip('▁')
    token = token.lstrip('')
    token = token.lstrip(' ')
    return token
def hg_strip_tokenizer_prefix(token):

    token = token.lstrip('▁')
    token = token.lstrip('')
    token = token.lstrip(' ')
    return token


def get_non_synonyms(word):
    """Fetch non-synonyms for a given word."""
    synonyms = {lemma.name() for syn in wn.synsets(word) for lemma in syn.lemmas()}
    non_synonyms = list(english_words - synonyms - {word})
    return non_synonyms if non_synonyms else [word]

def check_similarity(model_s, text1, text2, threshold=0.7):
    """Check if semantic similarity between two texts is below a threshold."""
    embedding1 = model_s.encode(text1, convert_to_tensor=True)
    embedding2 = model_s.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity < threshold

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


def calculate_word_scores(model_input, data):
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
    norma_attr_res = np.sum(absolute_attr_res, axis=0)
    normalized_attr_res = norma_attr_res / np.sum(norma_attr_res)
    tokens_attr_dict = generate_original_tokens_level_attribute(normalized_attr_res)
    return tokens_attr_dict


def calculate_component_scores(scored_tokens, component_positions_dict):
    # Convert token positions to a dictionary for faster lookup
    position_value_map = {token['position']: token for token in scored_tokens}

    # Initialize dictionaries for token and score storage
    components_tokens_dict = {}
    combined_scores_by_component = {}
    combined_scores_by_word = {}

    # Compute scores and tokens for each component
    for component, positions in component_positions_dict.items():
        component_tokens = [position_value_map.get(position) for position in positions if
                            position in position_value_map]
        components_tokens_dict[component] = component_tokens
        combined_scores_by_component[component] = sum(
            float(token['value']) for token in component_tokens if token is not None)
        combined_scores_by_word[component] = sum(
            float(position_value_map.get(position, {}).get('value', 0)) for position in positions)
    return combined_scores_by_component, combined_scores_by_word, components_tokens_dict


def calculate_component(component_sentences, word_scores):
    component_positions_dict = component_sentences

    combined_scores_by_component, combined_scores_by_word, components_tokens_dict = calculate_component_scores(
        word_scores,
        component_positions_dict)

    return_data = {'combined_scores_by_component': combined_scores_by_component,
                   'combined_scores_by_word': combined_scores_by_word,
                   'component_tokens_dict': components_tokens_dict}
    return return_data


def postproces_inferenced(df):
    def join_token_texts(component_dict, key):
        print(f"component_dict{component_dict}")
        tokens_list = component_dict[2][key]
        return tokens_list

    # Applying the function to each row for both 'instructions' and 'query'
    df['instructions_tokens'] = df['component_level'].apply(join_token_texts, key='instruction')
    df['query_tokens'] = df['component_level'].apply(join_token_texts, key='query')

    return df

def do_peturbed_reconstruct(df, modification_types=None):
    if modification_types is None:
        modification_types = ['top', 'bottom']

    # Helper function to tokenize and reconstruct prompts
    def reconstruct_prompt(row, modified_column, original_column='prompt'):
        modified_tokens = row[modified_column]
        tokenized_original = word_tokenize(row[original_column])
        updated_tokens = {
            mod_token['position']: mod_token['token']
            for mod_token in modified_tokens
            if 'position' in mod_token and 'token' in mod_token
        }
        return ' '.join([updated_tokens.get(i, token) for i, token in enumerate(tokenized_original)])

    # Extracting the unique percentage levels from column names
    percentage_levels = {float(re.search(r'_(\d+\.\d+)_', col).group(1)) for col in df.columns if
                         re.search(r'_\d+\.\d+_', col)}

    for mod_type in modification_types:
        for token_type in ['instruction', 'query']:
            for pct in percentage_levels:
                perturbed_col_pattern = f'{token_type}_token_{mod_type}_{pct}_peturbed'
                reconstructed_col_name = f'{mod_type}_reconstructed_{token_type}_{pct}'

                # Apply the reconstruct function only if the perturbed column exists
                if perturbed_col_pattern in df.columns:
                    df[reconstructed_col_name] = df.apply(reconstruct_prompt, axis=1,
                                                          modified_column=perturbed_col_pattern)

    return df


def old_do_peturbed_reconstruct(df, modification_types=None):
    def reconstruct_modified_prompt(original_prompt, modified_tokens):
        # Tokenize the original prompt to match positions
        tokenized_prompt = word_tokenize(original_prompt)
        position_to_token = {i: token for i, token in enumerate(tokenized_prompt)}

        # Update the mapping with modified tokens based on their positions
        for mod_token in modified_tokens:
            if 'position' in mod_token and 'token' in mod_token:
                position_to_token[mod_token['position']] = mod_token['token']

        # Reconstruct the prompt from the updated mapping
        reconstructed_prompt = [position_to_token[pos] for pos in sorted(position_to_token)]
        return ' '.join(reconstructed_prompt)

    # Identifying the unique percentage levels in the DataFrame columns
    percentage_levels = set()
    for column in df.columns:
        if 'peturbed' in column:
            try:
                # Extracting the percentage level from column name
                pct_level = column.split('_')[3]
                # Ensuring the extracted percentage is a float
                if pct_level.replace('.', '', 1).isdigit():
                    percentage_levels.add(float(pct_level))
            except IndexError:
                continue  # Skip columns that don't fit the expected naming convention

    for mod_type in modification_types:
        for token_type in ['instruction', 'query']:
            for pct in percentage_levels:
                column_name = f'{token_type}_token_{mod_type}_{pct}_peturbed'
                new_column_name = f'{mod_type}_reconstructed_{token_type}_{pct}'

                for idx, row in df.iterrows():
                    if column_name in df.columns:  # Check if this column exists
                        original_prompt = row['prompt']
                        modified_tokens = row[column_name]
                        modified_prompt = reconstruct_modified_prompt(original_prompt, modified_tokens)
                        df.at[idx, new_column_name] = modified_prompt

    return df

def modify_tokens(tokens, model_s, is_top, threshold=0.7, max_attempts=5, peturbation_level=0.4):
    original_text = ' '.join([t['token'] for t in tokens])
    tokens_sorted_by_value = sorted(tokens, key=lambda x: x['value'], reverse=True)

    # Calculate the slice index for top or bottom 40%
    num_tokens = len(tokens)
    slice_index = int(num_tokens * peturbation_level)

    # Determine tokens to be modified based on is_top flag
    if is_top:
        tokens_to_modify = tokens_sorted_by_value[:slice_index]
    else:
        tokens_to_modify = tokens_sorted_by_value[-slice_index:]

    # Keep the untouched segment as is
    untouched_tokens = tokens_sorted_by_value[slice_index:] if is_top else tokens_sorted_by_value[:-slice_index]

    for attempt in range(max_attempts):
        # Randomly replace tokens in the selected segment
        for token_info in tokens_to_modify:
            non_synonyms = get_non_synonyms(token_info['token'])
            token_info['token'] = random.choice(non_synonyms) if non_synonyms else token_info['token']

        # Reconstruct the text to check similarity
        modified_tokens = (tokens_to_modify + untouched_tokens) if is_top else (untouched_tokens + tokens_to_modify)
        modified_tokens = sorted(modified_tokens, key=lambda x: x['position'])  # Sort by original position
        modified_text = ' '.join([t['token'] for t in modified_tokens])

        # Check if modified text meets similarity threshold
        if check_similarity(model_s, original_text, modified_text, threshold):
            return modified_tokens  # Accept modifications if similarity is below threshold

    # If unable to meet threshold after attempts, enforce random changes
    for token_info in tokens_to_modify:
        token_info['token'] = random.choice(list(english_words))  # Force change with a random word

    return sorted(tokens_to_modify + untouched_tokens, key=lambda x: x['position'])
def run_peturbation(df):

    # Load the sentence model on GPU
    model_s = SentenceTransformer('all-MiniLM-L6-v2')

    # Make a deep copy of required columns to avoid modifying the original DataFrame.
    _df = copy.deepcopy(df[['instructions_tokens', 'query_tokens']])

    for pct in [0.2]:  # [0.1, 0.2, 0.3, 0.4]:
        df[f'instruction_token_top_{pct}_peturbed'] = _df['instructions_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=True, peturbation_level=pct)
        )
        df[f'instruction_token_bottom_{pct}_peturbed'] = _df['instructions_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=False, peturbation_level=pct)
        )
        df[f'query_token_top_{pct}_peturbed'] = _df['query_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=True, peturbation_level=pct)
        )
        df[f'query_token_bottom_{pct}_peturbed'] = _df['query_tokens'].apply(
            lambda lst: modify_tokens(copy.deepcopy(lst), model_s, is_top=False, peturbation_level=pct)
        )


    return df

