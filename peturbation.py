import copy
import random
import torch
import os
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from sentence_transformers import SentenceTransformer, util
import re
from nltk.tokenize import word_tokenize
import logging

# Load the set of English words from NLTK
english_words = set(words.words())


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
        # if check_similarity(model_s, original_text, modified_text, threshold):
        #     return modified_tokens  # Accept modifications if similarity is below threshold

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


