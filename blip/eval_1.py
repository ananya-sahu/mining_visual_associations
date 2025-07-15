import os
import sys
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BlipProcessor, BlipForImageTextRetrieval

# --- Wrapper (your existing class, slightly adjusted for evaluation) ---
class Blip1Wrapper(nn.Module):
    def __init__(self, blip_model, num_tasks, num_embeds, embed_dim, method='third'):
        super().__init__()
        self.vision_model = blip_model.vision_model
        self.text_encoder = blip_model.text_encoder
        self.tokenizer = blip_model.text_encoder.embeddings

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

# --- Dataset Loader ---
def load_dataset(json_file, img_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    samples = []
    for img_path, entries in data.items():
        filename = os.path.basename(img_path)
        full_img_path = os.path.join(img_dir, filename)
        for task_label, caption in entries:
            samples.append((full_img_path, caption, int(task_label) - 1))
    return samples

# --- Similarity Computation ---
def get_similarity(samples, processor, tokenizer, device, model, task=None, baseline=True, batch_size=64):
    image_paths = [s[0] for s in samples]
    captions = [s[1] for s in samples]
    task_ids = [task if task is not None else s[2] for s in samples]

    num = len(samples)
    similarity_matrix = torch.zeros((num, num), device=device)

    with torch.no_grad():
        for start in tqdm(range(0, num, batch_size), desc="Computing similarity"):
            end = min(start + batch_size, num)
            img_batch = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
            cap_batch = captions[start:end]
            task_batch = torch.tensor(task_ids[start:end], device=device)

            pixel_values = processor(images=img_batch, return_tensors="pt")["pixel_values"].to(device)
            tokenized = tokenizer(
                cap_batch,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)

            if baseline:
                image_feats = model.vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
                text_feats = model.text_encoder(input_ids=tokenized["input_ids"],attention_mask=tokenized["attention_mask"]).last_hidden_state[:, 0, :]
                image_feats = F.normalize(image_feats, dim=-1)
                text_feats = F.normalize(text_feats, dim=-1)
            else:
                image_feats, text_feats = model(pixel_values, tokenized["input_ids"], tokenized["attention_mask"], task_batch)

            similarity_matrix[start:end, start:end] = text_feats @ image_feats.T

    return similarity_matrix

# --- Metrics ---
def recall_at_k(sim, k):
    correct = 0
    for i in range(sim.shape[0]):
        if i in torch.topk(sim[i], k).indices:
            correct += 1
    return correct / sim.shape[0]

def average_rank(sim):
    total_rank = 0
    for i in range(sim.shape[0]):
        sorted_idx = torch.argsort(sim[i], descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
        total_rank += rank
    return total_rank / sim.shape[0]

# --- Main ---
def main():
    baseline_flag = sys.argv[1].lower() == "true"  # "true" or "false"
    checkpoint = None if sys.argv[2].lower() == "none" else sys.argv[2]
    json_file = sys.argv[3]
    img_dir = sys.argv[4]
    task = None if sys.argv[5].lower() == "none" else int(sys.argv[5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_base = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)

    if baseline_flag:
        model = model_base
    else:
        model = Blip1Wrapper(
            blip_model=model_base,
            num_tasks=5,
            num_embeds=5,
            embed_dim=model_base.vision_model.config.hidden_size
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

    samples = load_dataset(json_file, img_dir)
    similarity_matrix = get_similarity(samples, processor.image_processor, processor.tokenizer, device, model, task=task, baseline=baseline_flag)

    print("Recall@1:", recall_at_k(similarity_matrix, 1))
    print("Recall@5:", recall_at_k(similarity_matrix, 5))
    print("Recall@10:", recall_at_k(similarity_matrix, 10))
    print("Recall@20:", recall_at_k(similarity_matrix, 20))
    print("Avg Rank:", average_rank(similarity_matrix))

if __name__ == "__main__":
    # Example run:
    # sys.argv = [
    #     "evaluate_blip.py",
    #     "true",                  # baseline or not
    #     "/content/blip_model.pt",     # checkpoint if baseline=False
    #     "dummy_data.json",        # json file
    #     "dummy_eval_imgs",              # image folder
    #     "none"                    # task or "none"
    # ]
    main()