import torch
import os
from PIL import Image
from tqdm import tqdm
import sys
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch.nn.functional as F
from torch import nn

# ---- Load Task-Aware BLIP1 Wrapper ----
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

# --- Utility ---
def extract_texts_and_images(root_dirs, image_extensions={".jpg", ".jpeg", ".png", ".bmp", ".gif"}):
    images_to_text = {}
    for root_dir in root_dirs:
        for text_label in os.listdir(root_dir):
            text_path = os.path.join(root_dir, text_label)
            if os.path.isdir(text_path):
                for file in os.listdir(text_path):
                    file_path = os.path.join(text_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(tuple(image_extensions)):
                        images_to_text[file_path] = text_label
    return images_to_text

def compute_similarity(images_to_text, model, task_num, processor, device, batch_size=512, base=False):
    texts = list(images_to_text.values())
    image_paths = list(images_to_text.keys())

    num_samples = len(texts)
    image_features_list, text_features_list = [], []

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size)):
            end = min(start + batch_size, num_samples)
            text_batch = texts[start:end]
            image_batch = image_paths[start:end]

            images = [Image.open(p).convert("RGB") for p in image_batch]
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]

            tokenized = processor.tokenizer(
                text_batch,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            tokenized = {k: v for k, v in tokenized.items() if k in {"input_ids", "attention_mask"}}

            inputs = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "pixel_values": pixel_values
            }

            for key in inputs:
                inputs[key] = inputs[key].to(device)

            if base:
                image_feats = model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0, :]
                text_feats = model.text_encoder(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"]).last_hidden_state[:, 0, :]
                image_feats = F.normalize(image_feats, dim=-1)
                text_feats = F.normalize(text_feats, dim=-1)
            else:
                task_ids = torch.full((inputs["pixel_values"].size(0),), task_num, device=device)
                image_feats, text_feats = model(
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    task_ids
                )

            image_feats = F.normalize(image_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)
            image_features_list.append(image_feats)
            text_features_list.append(text_feats)

    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    return image_features @ text_features.T

# --- Metrics ---
def recall_at_k(similarity_matrix, k):
    recalls = 0
    for i in range(similarity_matrix.shape[0]):
        top_k = torch.topk(similarity_matrix[i], k=k).indices
        if i in top_k:
            recalls += 1
    return recalls / similarity_matrix.shape[0]

def average_rank(similarity_matrix):
    total_rank = 0
    for i in range(similarity_matrix.shape[0]):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        total_rank += rank
    return total_rank / similarity_matrix.shape[0]

# --- Main Evaluation Loop ---
def get_results(paths, model, processor, device, dists=None, base=False):
    mapping = extract_texts_and_images(paths)

    if base:
        sim = compute_similarity(mapping, model, task_num=0, processor=processor, device=device, base=True)
        print("Average Rank:", average_rank(sim))
    else:
        for dist in dists:
            print(f"\n--- Distance Level: {dist} ---")
            sim = compute_similarity(mapping, model, task_num=dist, processor=processor, device=device, base=False)
            print("Recall@1:", recall_at_k(sim, 1))
            print("Recall@5:", recall_at_k(sim, 5))
            print("Recall@10:", recall_at_k(sim, 10))
            print("Recall@20:", recall_at_k(sim, 20))
            print("Average Rank:", average_rank(sim))

# --- Entry Point ---
def main():
    baseline = sys.argv[1].lower() == "true"
    checkpoint = sys.argv[2] if sys.argv[2].lower() != "none" else None
    root_dirs = sys.argv[3:9]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_base = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)

    if baseline:
        get_results(root_dirs, model_base, processor, device, base=True)
    else:
        wrapper = Blip1Wrapper(
            blip_model=model_base,
            num_tasks=5,
            num_embeds=5,
            embed_dim=model_base.vision_model.config.hidden_size
        ).to(device)

        wrapper.load_state_dict(torch.load(checkpoint, map_location=device))
        get_results(root_dirs, wrapper, processor, device, dists=[0, 1, 2, 3, 4], base=False)

if __name__ == "__main__":
    # sys.argv = [
    #     "evaluate_blip1_metaphor.py",
    #     "true",                   # baseline ("true" or "false")
    #     "none",  # checkpoint or "none"
    #     "dummy_data",
    #     "dummy_data/label_0",
    #     "dummy_data/label_1",
    #     "dummy_data/label_2"
    # ]
    main()