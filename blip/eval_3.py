import torch
import os
import pickle
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForImageTextRetrieval
from torch import nn
import sys

class Blip1Wrapper(nn.Module):
    def __init__(self, blip_model, num_tasks, num_embeds, embed_dim, method='third'):
        super().__init__()
        self.vision_model = blip_model.vision_model
        self.text_encoder = blip_model.text_encoder
        self.tokenizer = blip_model.text_encoder.embeddings  # not used directly
        self.task_embeddings = nn.ModuleList([
            nn.Embedding(num_tasks, embed_dim) for _ in range(num_embeds)
        ])
        self.method = method
        self.project = nn.Linear(embed_dim * 2, embed_dim)

    def encode_image(self, pixel_values, task_ids):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state  # (B, N+1, D)
        task_embeds = torch.stack([e(task_ids) for e in self.task_embeddings], dim=1)  # (B, E, D)

        if self.method == 'first':
            x = torch.cat((task_embeds, patch_tokens), dim=1)
        elif self.method == 'second':
            cls_token = patch_tokens[:, :1, :]
            patches = patch_tokens[:, 1:, :]
            x = torch.cat([cls_token, task_embeds, patches], dim=1)
        elif self.method == 'third':
            x = torch.cat((patch_tokens, task_embeds), dim=1)
        elif self.method == 'concat':
            pooled = patch_tokens[:, 0, :]
            task_embed_mean = task_embeds.mean(dim=1)
            x = torch.cat([pooled, task_embed_mean], dim=-1)
            return F.normalize(self.project(x), dim=-1)
        else:
            x = patch_tokens

        pooled = x[:, 0, :]
        return F.normalize(pooled, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return F.normalize(pooled, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask, task_ids):
        image_features = self.encode_image(pixel_values, task_ids)
        text_features = self.encode_text(input_ids, attention_mask)
        return image_features, text_features

def compute_similarity(data, processor, model, device, task=None, baseline=True):
    sim_maxes = []
    
    for text, entry in data.items():
        image_paths = [entry["DALLE"], entry["DALLE_CoT"]]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        tokenized = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            if baseline:
                image_feats = model.vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
                text_feats = model.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:, 0, :]
                image_feats = F.normalize(image_feats, dim=-1)
                text_feats = F.normalize(text_feats, dim=-1)
            else:
                task_ids = torch.full((2,), task, dtype=torch.long, device=device)
                image_feats, text_feats = model.forward(pixel_values, input_ids, attention_mask, task_ids)
            sims = (text_feats @ image_feats.T)[0]
            sim_maxes.append(torch.argmax(sims).item())

    return np.mean(sim_maxes)

def main():
    checkpoint = sys.argv[1]
    baseline = sys.argv[2].lower() == "true"
    data_file = sys.argv[3]

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_base = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)

    if baseline:
        result = compute_similarity(data, processor, model_base, device, baseline=True)
        print(f"Baseline accuracy: {result:.4f}")
    else:
        wrapper = Blip1Wrapper(
            blip_model=model_base,
            num_tasks=5,
            num_embeds=5,
            embed_dim=model_base.vision_model.config.hidden_size,
        ).to(device)

        wrapper.load_state_dict(torch.load(checkpoint, map_location=device))
        for task in range(5):
            result = compute_similarity(data, processor, wrapper, device, task=task, baseline=False)
            print(f"Distance {task} accuracy: {result:.4f}")

if __name__ == "__main__":
#     sys.argv = [
#       "evaluate_blip1_contrastive.py",    # fake script name placeholder
#       "none",           # path to your checkpoint OR "none" if baseline
#       "true",                            # "true" for baseline BLIP, "false" to use task-augmented wrapper
#       "/content/dummy_contrastive.pkl"    # path to your pickle file: {text: {"DALLE": path, "DALLE_CoT": path}}
#   ]

    main()