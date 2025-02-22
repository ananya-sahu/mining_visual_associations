import pickle 
import torch 
import numpy as np
from PIL import Image,UnidentifiedImageError
import json
from collections import defaultdict
from tqdm import tqdm
import spacy
import pandas as pd
import random 
import sys
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}

        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

def load_coco_captions(file_path):
  """
  Args: json file path, returns dictionary
  """
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

def load_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_data_point(long_caps, dense_cap_path, image_id, split):
  full_path = f"""{dense_cap_path}/{split}/COCO_{split}_{str(image_id).zfill(12)}.jpg"""
  dense_cap = long_caps[full_path][0]
  return dense_cap

def get_meaningful_words(sentence):
    sentence = sentence.lower()
    doc = nlp(sentence)
    meaningful_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop]
    return set(meaningful_words)

def get_all_meaningful_words(lemmatizer, ratings, image_ids,num_words):
    meaningful_words = defaultdict(list)
    for image_id in tqdm(image_ids):
        short_caption_item = random.choice(image_ids[image_id])
        id = short_caption_item['id']
        caption = short_caption_item['caption']
        list_words = get_meaningful_words(caption)
        by_rating = {}
        for word in list_words:
            lemmatized_word = lemmatizer.lemmatize(word.lower())
            if (lemmatized_word in ratings):
                by_rating[word] = ratings[lemmatized_word]['Conc.M']
        words_sorted = sorted(by_rating, key=lambda x: by_rating[x], reverse=True)
        if len(words_sorted) > num_words:
            words_sorted = words_sorted[:num_words]

        meaningful_words[image_id] = (caption, words_sorted, id)
        
    return meaningful_words

def get_system_prompt(context_caption,short_caption):
    prompt = f"""For a given list of words, generate a new list for each word using the same part of speech. The words should follow a semantic abstraction scale where distance increases from near-synonyms to abstract concepts. 
    Approach:
    1. Distance 1 – Near Synonyms: Close in meaning or form (e.g., Ball → Sphere). 
    2. Distance 2 – Slight Abstraction: Slightly broader category (e.g., Ball → Toy).
    3. Distance 3 – Broader Context: Indirect, but linked through situational and emotional context (e.g., Ball → Game).
    4. Distance 4 – Conceptual Association: More abstract or theme-related (e.g., Ball → Competition).
    5. Distance 5 – Full Abstraction: Highly abstract or metaphorical (e.g., Ball → Journey).
    Generate three words each for distances 1 to 5. Generated words should fit into the overall emotional and situational context of this context caption: `{context_caption}`. Generated words when replaced with the original word in this short caption {short_caption} should semantically be correct. 
    Do not generate the original word in the new generations. Use JSON format: the key is the original word, the value is a dictionary with distances as keys and the list of generated words as values."""
    return prompt

def get_prompts(image_ids, salient_words, split,long_caps):
    tasks = []
    for key in image_ids:
        dense_caption = get_data_point(long_caps,key, split)
        short_caption = salient_words[key][0]
        words = str(salient_words[key][1])
        prompt = get_system_prompt(dense_caption,short_caption)
        custom_id = f"""imageid_{key}_captionid_{salient_words[key][2]}"""
        
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "response_format": { 
                    "type": "json_object"
                },
                "messages": [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": words
                    }
                ],
                "max_tokens": 1000
            }
        }
        
        tasks.append(task)
    return tasks


def main():
    # Load captions for training and validation sets
    train_captions = sys.argv[1]
    val_captions = sys.argv[2]
    train_annotations = sys.argv[3]
    images_dir = sys.argv[4]
    concretness_ratings = sys.argv[5]
    dense_descriptions = sys.argv[6]
    train_save_path = sys.argv[7]
    val_save_path = sys.argv[8]
    train_captions = load_coco_captions(train_captions)
    val_captions = load_coco_captions(val_captions)
    coco_annotations_file = train_annotations
    coco_images_dir = images_dir
    coco= COCOParser(coco_annotations_file, coco_images_dir)

    #format train caption where we have image id: then the 5 captions that belong to it 
    train_captions_by_image_id = defaultdict(list)
    for annotation in train_captions['annotations']:
        train_captions_by_image_id[annotation['image_id']].append(annotation)
    val_captions_by_image_id = defaultdict(list)
    for annotation in val_captions['annotations']:
        val_captions_by_image_id[annotation['image_id']].append(annotation)

    ratings = pd.read_csv(concretness_ratings,index_col=False)
    ratings = ratings.set_index('Word').to_dict(orient='index')
    long_caps = load_json_to_dict(dense_descriptions)
    lemmatizer = WordNetLemmatizer()
    salient_words_train = get_all_meaningful_words(lemmatizer, train_captions_by_image_id, 5)
    salient_words_val = get_all_meaningful_words(lemmatizer, val_captions_by_image_id,5)
    train_prompts = get_prompts(train_captions_by_image_id, salient_words_train, "train2014",long_caps)
    val_prompts = get_prompts(val_captions_by_image_id, salient_words_val,"val2014",long_caps)

    with open(train_save_path, 'w') as f:
        for entry in train_prompts:
            f.write(json.dumps(entry) + '\n')

    with open(val_save_path, 'w') as f:
        for entry in val_prompts:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    main()