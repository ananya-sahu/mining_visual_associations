from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
import os
import pickle
import gc
import torch
import sys

def chunk_data(input_dict, batch_size):
    # Convert the dictionary into a list of key-value pairs
    items = list(input_dict.items())
    # Create batches of key-value pairs
    for i in range(0, len(items), batch_size):
        yield dict(items[i:i + batch_size])

def process_batch(prompts_batch, llm, sampling_params):
    inputs = []
    results = {}
    ids = []
    for idx, (label, image_path, new_word, salient_word, prompt) in prompts_batch.items():
            with Image.open(image_path) as img:
                img.load()
                ids.append(idx)
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img
                    }
                })
        
    try:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        for i, result in enumerate(outputs):
            idx = ids[i]
            results[idx] = result.outputs[0].text if result.outputs else {"error": "No output"}
    except Exception as e:
        print(e)

    return results

def save_results(results, file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
            results.update(existing_data)

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

# Main processing loop
def main():
    start_batch_idx = 0 #change this if needed
    raw_data_file_path = sys.argv[1]
    split = sys.argv[2]
    label = sys.argv[3]
    enviorn_path = sys.argv[4]
    save_data_path = sys.arg[5]
    # Set cache environment
    os.environ['HF_HOME'] = enviorn_path

    # Load model
    model_name = "allenai/Molmo-7B-D-0924"
    llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16")
    #load data 
    with open(str(raw_data_file_path), "rb") as f:
        data = pickle.load(f)
    #format of data is id as key and value as label (int),image_path,new_word,salient_word, prompt (all else strings)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=150, n=1
    )

    batch_size = 1000
    batches = list(chunk_data(data, batch_size))
    num_batches = len(batches)
    try:
        for batch_idx in tqdm(range(0,num_batches)):
            batch_results = process_batch(batches[batch_idx], llm, sampling_params)
            # Save after each batch
            save_idx = batch_idx + start_batch_idx
            save_path = f"{save_data_path}/{split}/{label}/{split}_{save_idx}.pkl"
            save_results(batch_results,save_path)
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
            print(e,batch_idx)

if __name__ == "__main__":
    main()





