import json
import glob
import sys
import re
import random
import pickle
from collections import defaultdict

#after getting associations from gpt4o, format into one list of associations
def read_jsonl_files(directory):
  """Reads all JSONL files in a directory and returns a list of dictionaries.

  Args:
    directory: The path to the directory containing the JSONL files.

  Returns:
    A list of dictionaries, where each dictionary represents a JSON object from
    a JSONL file. Returns an empty list if no JSONL files are found or if there
    are errors reading the files.
  """
  all_data = []
  for filename in glob.glob(f"{directory}/*.jsonl"):
    try:
      with open(filename, 'r') as f:
        for line in f:
          try:
            data = json.loads(line)
            all_data.append(data)
          except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
      print(f"An unexpected error occurred while reading {filename}: {e}")
  return all_data

def make_dictionary(input_string):
    #takes the gpt4o json format and restructures to a dictionary
    try:
        json_object = json.loads(input_string)
        return json_object
    except json.JSONDecodeError:
        pass 
    pattern = re.compile(r'"[\w\s]*\d": \[\s*("[^"]+",\s*)*"[^"]+"\s*\]', re.MULTILINE)
    matches = list(pattern.finditer(input_string))
    if not matches:
        print(input_string)
        return "No valid JSON structure found."
    last_valid_position = matches[-1].end()
    truncated_string = input_string[:last_valid_position]
    open_brackets = truncated_string.count("{")
    close_brackets = truncated_string.count("}")
    truncated_string += "}" * (open_brackets - close_brackets)
    try:
        json_object = json.loads(truncated_string)
        return json_object
    except json.JSONDecodeError:
        return "Failed to create a valid JSON structure."

def get_formatted_data(data):
    all_data = {}
    for e in data:
        d_str = make_dictionary(e['response']['body']['choices'][0]['message']['content'])
        if d_str is not None:
            all_data[e['custom_id']] = d_str
    return all_data

def load_coco_captions(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

def get_short_captions(annotations, data_set):
  short_captions = {}
  grouped = defaultdict(defaultdict)
  for a in annotations['annotations']:
    grouped[str(a['image_id'])][str(a['id'])] = a['caption']

  for id in data_set:
     _,image_id,_,caption_id = id.split("_")
     short_captions[id] = grouped[image_id][caption_id]


  return short_captions

def format_prompt(all_salient_words, salient_word, label, new_word): #this is to format into molmo prompts 
    try:
        words = new_word
        for word in all_salient_words:
            words += ", " + word
        all_words = words.replace(salient_word, "")
        prompt = f"USER: <image> \n Write a short caption that is grounded in this image and semantically correct in less that 10 words choosing some or all of these words: {all_words} to make the caption best represent the image. Steer the style of the caption toward this distance level of abstraction {str(label)} following these rules: distance 1 – Near Synonyms: Close in meaning or form of the original image, Distance 2 – Slight Abstraction: Slightly more loose and abstract to the image, Distance 3 – Broader Context: Indirect, but linked through situational and emotional context of the image, Distance 4 – Conceptual Association: More abstract or theme-related to the image, Distance 5 – Full Abstraction: Highly abstract or metaphorical to the image. The caption MUST contain {new_word}. \n ASSISTANT:"
        return prompt
    except Exception as e:
        return f"Error formatting prompt: {str(e)}"

def get_prompt_parts(main_path, data, shorts, data_spec):
    label_list = []
    image_list = []
    original_captions = []
    salient_words = []
    all_salient_words = []
    new_words = []
    error_keys = []

    for key in data:
        try:
            parts = key.split('_')
            image_id = parts[1]
            first_part = f"{main_path}/coco_{data_spec}/{data_spec}"
            image = f"{first_part}/{str(image_id).zfill(12)}.jpg"
            caption = shorts[key]
            for word in data[key]:
                for dist in data[key][word]:
                    numerical_dist = re.findall(r'\b\d\b', dist)
                    if len(numerical_dist) > 0:
                        if isinstance(data[key][word][dist], list):
                            for new_word in [random.choice(data[key][word][dist])]: #note here we are only picking on of the associations for each word at each 
                                                                                    # distance but to do all just need to iterate over all choices 
                                label_list.append(int(numerical_dist[0]))
                                image_list.append(image)
                                original_captions.append(caption)
                                salient_words.append(word)
                                new_words.append(new_word)
                                all_salient_words.append(list(data[key].keys()))
        except Exception as e:
            error_keys.append(key)
            print(f"Error processing key {key}: {str(e)}")

    return label_list, image_list, original_captions, salient_words, all_salient_words, new_words, error_keys

def get_all_prompts(label_list, image_list, salient_words, all_salient_words, new_words):
    prompts = {}
    error_ids = []

    for i in range(len(label_list)):
        try:
            # Prompt dict has label, image file name, prompt
            if "Error formatting prompt:" in format_prompt(all_salient_words[i], salient_words[i], label_list[i], new_words[i]):
                error_ids.append(i)
            prompts[i] = (
                label_list[i],
                image_list[i],
                new_words[i],
                salient_words[i],
                format_prompt(all_salient_words[i], salient_words[i], label_list[i], new_words[i])
            )
        except Exception as e:
            error_ids.append(i)
            print(f"Error generating prompt for index {i}: {str(e)}")

    return prompts, error_ids

def seperate_data(data):
    """
    args: all prompts dictionary
    returns: dictionary of prompts seperated by their distance labels
    """
    seperated_by_label = defaultdict(defaultdict)
    for ind in data:
        seperated_by_label[data[ind][0]][ind] = data[ind]
    return seperated_by_label

def save_by_label(split,data, loc_path):
    for i in range (1,6):
        save_path = f"{loc_path}/{split}/{split}_{i}.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(data[i], file)
def main():
  train_associations = sys.argv[1] #the directory containing all outputs from gpt4o for train set
  val_associations = sys.argv[2] #the directory containing all outputs from gpt4o for val set
  train_captions = sys.argv[3]
  val_captions = sys.argv[4]
  loc_path = sys.argv[5]
  train_data = read_jsonl_files(train_associations)
  val_data = read_jsonl_files(val_associations)

  train_formatted = get_formatted_data(train_data) #formatted into dictionaries 
  val_formatted = get_formatted_data(val_data)
  train_captions = load_coco_captions(train_captions)
  val_captions = load_coco_captions(val_captions)

  train_short = get_short_captions(train_captions, train_formatted)
  val_short = get_short_captions(val_captions, val_formatted)

  train_label_list, train_image_list, train_original_captions, train_salient_words, train_all_salient_words, train_new_words, train_errors = get_prompt_parts(
    train_formatted, train_short, "train2017")
  train_prompts, train_prompt_errors = get_all_prompts(train_label_list, train_image_list, train_salient_words, train_all_salient_words, train_new_words)
  val_label_list, val_image_list, val_original_captions, val_salient_words, val_all_salient_words, val_new_words, val_errors = get_prompt_parts(
      val_formatted, val_short, "val2017"
  )
  val_prompts, val_prompt_errors = get_all_prompts(val_label_list, val_image_list, val_salient_words, val_all_salient_words, val_new_words)

  train_seperated = seperate_data(train_prompts) 
  val_seperated = seperate_data(val_prompts)

  save_by_label(train_seperated,"train", loc_path)
  save_by_label(val_seperated,"val", loc_path)

if __name__ == "__main__":
   main()








  