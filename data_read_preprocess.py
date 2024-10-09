from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# global var



def preprocess_data(examples):
    instruction = """Analyze the sentiment of the previous sentence and respond only with ^POSITIVE^ or ^NEGATIVE^ Your answer is ^"""
    updated_sentences = []
    ranges = []
    queries = []
    instructions = []
    lengths = []
    for query in examples['sentence']:
        combined = query + ' ' + instruction
        updated_sentences.append(combined)
        range_info = get_query_instruction_positions(query)
        ranges.append(range_info)
        queries.append(query)
        instructions.append(instruction)
        lengths.append(len(query))
    examples['sentence'] = updated_sentences
    examples['component_range'] = ranges
    examples['instruction'] = instructions
    examples['query'] = queries
    examples['length'] = lengths
    #print(lengths)
    return examples


def get_query_instruction_positions(query):
    # Concatenate the sentences
    instruction = """Analyze the sentiment of the previous sentence and respond only with ^POSITIVE^ or ^NEGATIVE^ Your answer is ^"""
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
    dataset = load_dataset("sst2", split='train')
    preprocessed_dataset = dataset.map(preprocess_data, batched=True,  load_from_cache_file=False)
    preprocessed_dataset = preprocessed_dataset.sort('length')
    return preprocessed_dataset.select(range(dataset_range[0],dataset_range[1]-1))


if __name__ == "__main__":

    df = load_and_preprocess([0,63000])


    print(df[16000])
    print(df[24000])
    print(df[32000])
    print(df[40000])
    print(df[56000])

