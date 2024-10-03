from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from qa_preprocess import load_and_preprocess
import pandas as pd
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
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available resources
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
def generate_text_with_ig(model, tokenizer, current_input):
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
                             )
    response = tokenizer.decode(outputs['sequences'][0][len(inputs["input_ids"][0]):], skip_special_tokens=True)



    return response
def run_initial_inference(model,tokenizer,df):

    print(len(df))
    data = []

    for ind, example in enumerate(df.select(range(len(df)-1))):

            real_output = generate_text_with_ig(model, tokenizer,example['query'] )

            if real_output is not None:
                data.append(
                    {'prompt': example['query'], "real_output": real_output,

                     }
                )
            else:
                print(f"hg_infer.py:170  No output for prompt: {example['sentence']}")
    result = pd.DataFrame(data)

    return result
def main(model, tokenizer,df,start,end ):

   # method = "gradient"
    inference_df = run_initial_inference(model=model,tokenizer=tokenizer,df=df)
    inference_df.to_pickle(f"{start}_{end}_qa_hg_baseline_inferenced_df.pkl")
    print("\ndone the inference")


if __name__ == "__main__":
    model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf", BitsAndBytesConfig(bits=4, quantization_type="fp16"))
    start = 5103
    end = start + 200
    df = load_and_preprocess([start, end])

    #main("gradient")
    main(model, tokenizer,df,start,end)

