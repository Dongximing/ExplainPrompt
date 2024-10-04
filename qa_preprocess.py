from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# global var



def preprocess_data(examples):
    suffix = "give a short answer:"
    ranges = []
    lengths = []
    updated_sentences = []
    queries = []
    instructions = []
    for query in examples['question']:
        #query = f"<s>[INST] {query} [/INST]"
        combined = query + ' ' + suffix
        updated_sentences.append(combined)
        range_info = get_query_instruction_positions(query)
        ranges.append(range_info)
        queries.append(query)
        instructions.append(suffix)
        lengths.append(len(query))
    examples['prefix_query'] = updated_sentences
    examples['component_range'] = ranges
    examples['instruction'] = instructions
    examples['query'] = queries
    examples['length'] = lengths

    return examples
def preprocess_data_baseline(examples):

    lengths = []
    queries = []

    for query in examples['question']:
        queries.append(query)
        lengths.append(len(query))

    examples['query'] = queries
    examples['length'] = lengths

    return examples


def get_query_instruction_positions(query):
    # Concatenate the sentences
    instruction = "give a short answer:"
    #query = f"<s>[INST] {query} [/INST]"
    combined_text = query + ' ' + instruction

    # Tokenize the combined text
    total_range = len(word_tokenize(combined_text))

    # find count of first
    query_end = len(word_tokenize(query))

    return {"query": list(range(0, query_end)),
            "instruction": list(range(query_end, total_range))
            }

def sort_by_length(example):
    return example['length']

def load_and_preprocess(dataset_range):
    dataset = load_dataset("rajpurkar/squad_v2", split='train')
    preprocessed_dataset = dataset.map(preprocess_data, batched=True,  load_from_cache_file=False)
    preprocessed_dataset = preprocessed_dataset.sort('length')
    return preprocessed_dataset.select(range(dataset_range[0],dataset_range[1]-1))

def load_and_preprocess_baseline(dataset_range):
    dataset = load_dataset("rajpurkar/squad_v2", split='train')
    preprocessed_dataset = dataset.map(preprocess_data_baseline, batched=True,  load_from_cache_file=False)
    preprocessed_dataset = preprocessed_dataset.sort('length')
    return preprocessed_dataset.select(range(dataset_range[0],dataset_range[1]-1))
if __name__ == "__main__":

    df = load_and_preprocess([5000,6000])
    print(df[10])
    print(df[11])
    print(df[12])
