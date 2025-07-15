import torch
import os
import pickle
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import SiglipProcessor, SiglipModel
from torch import nn
import sys

class SigLIPWrapper(nn.Module):
    def __init__(self, siglip_model, num_tasks, num_embeds, embed_dim, method='third'):
        super().__init__()
        self.siglip_model = siglip_model
        self.task_embeddings = nn.ModuleList([nn.Embedding(num_tasks, embed_dim) for _ in range(num_embeds)])
        self.method = method

    def forward_image(self, images, task_ids):
        image_embeds = self.siglip_model.vision_model(pixel_values=images).last_hidden_state  # [B, P, D]
        task_embeds = torch.stack([embed(task_ids) for embed in self.task_embeddings], dim=1)

        if self.method == 'first':
            x = torch.cat((task_embeds, image_embeds), dim=1)
        elif self.method == 'second':
            cls_token, patches = image_embeds[:, :1, :], image_embeds[:, 1:, :]
            x = torch.cat((cls_token, task_embeds, patches), dim=1)
        else:  # 'third'
            x = torch.cat((image_embeds, task_embeds), dim=1)

        pooled = self.siglip_model.vision_model.post_layernorm(x)
        pooled = pooled.mean(dim=1)  # simple average pool
        return F.normalize(pooled, dim=-1)

    def forward_text(self, input_ids, attention_mask):
        output = self.siglip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return F.normalize(output.pooler_output, dim=-1)

def compute_similarity(data, processor, model, device, task=None, baseline=True):
    sim_maxes = []
    
    for text, entry in data.items():
        # Load and preprocess images
        image_paths = [entry["DALLE"],entry["DALLE_CoT"]]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        pixel_values = processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        
        # Tokenize text
        tokenized = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
            return_attention_mask=True
        )
        inputs = {
            "input_ids": tokenized["input_ids"].to(device),
            "attention_mask": tokenized["attention_mask"].to(device),
            "pixel_values": pixel_values
        }
        
        with torch.no_grad():
            if baseline:
                image_feats = model.get_image_features(pixel_values=inputs["pixel_values"])
                text_feats = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            else:
                task_ids = torch.full((2,), task, device=device, dtype=torch.long)
                image_feats = model.forward_image(inputs["pixel_values"], task_ids)
                text_feats = model.forward_text(inputs["input_ids"], inputs["attention_mask"])

            # Compute similarity between text and images
            sims = (text_feats @ image_feats.T)[0]
            sim_maxes.append(torch.argmax(sims).item())

    return np.mean(sim_maxes)


def main():
    checkpoint = sys.argv[1]
    baseline = sys.argv[2].lower() == "true"
    data_file = sys.argv[3]

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    model_base = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)

    if baseline:
        result = compute_similarity(data, processor, model_base, device, baseline=True)
        print(f"Baseline accuracy: {result:.4f}")
    else:
        wrapper = SigLIPWrapper(
            siglip_model=model_base,
            num_tasks=5,
            num_embeds=5,
            embed_dim=model_base.config.vision_config.hidden_size
        ).to(device)

        wrapper.load_state_dict(torch.load(checkpoint, map_location=device))
        for task in range(5):
            result = compute_similarity(data, processor, wrapper, device, task=task, baseline=False)
            print(f"Distance {task} accuracy: {result:.4f}")

if __name__ == "__main__":
    # sys.argv = [
    # "evaluate_siglip_contrastive.py",
    # "/content/siglip_model.pt",  # path to checkpoint
    # "false",                     # "true" for baseline SigLIP, "false" for your wrapper model
    # "/content/dummy_contrastive.pkl"          # pickle file containing {text: {"DALLE": path, "DALLE_CoT": path}} structure
    # ]
    main()