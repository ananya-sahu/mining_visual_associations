import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from transformers import BlipProcessor, BlipForImageTextRetrieval

# --- Task-Balanced Sampling ---
def get_data_balanced(samples):
    reorg = defaultdict(lambda: defaultdict(list))
    for img_path, caption, label in samples:
        reorg[img_path][label].append((img_path, caption, label))
    balanced = []
    while True:
        added = False
        for img in reorg:
            for label in reorg[img]:
                if reorg[img][label]:
                    balanced.append(reorg[img][label].pop(0))
                    added = True
        if not added:
            break
    return balanced

# --- Dataset ---
class JsonCaptionDataset(Dataset):
    def __init__(self, json_path, home_dir, image_processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.samples = []
        for img_path, entries in self.data.items():
            for task_label, caption in entries:
                filename = os.path.basename(img_path)
                full_path = os.path.join(home_dir, filename)
                self.samples.append((full_path, caption, int(task_label) - 1))
        self.image_processor = image_processor

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, task_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return image, caption, torch.tensor(task_label, dtype=torch.long)

# --- Collate ---
def collate_fn(batch, tokenizer, max_length=64):
    images, captions, task_labels = zip(*batch)
    images = torch.stack(images)
    tokenized = tokenizer(list(captions), padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "pixel_values": images,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "task_ids": torch.stack(task_labels)
    }

# --- BLIP-1 Dual Encoder with Task Prefix ---
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

# --- InfoNCE Contrastive Loss ---
def compute_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = image_features @ text_features.T / temperature
    logits_per_text = text_features @ image_features.T / temperature

    labels = torch.arange(len(image_features), device=image_features.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2

# --- Recall@K ---
def recall_at_k(sim_matrix, k):
    correct = 0
    for i in range(sim_matrix.size(0)):
        if i in torch.topk(sim_matrix[i], k=k).indices:
            correct += 1
    return correct / sim_matrix.size(0)

# --- Training ---
def train_with_val(train_json, val_json, train_dir, val_dir, save_path,
                   num_workers=4, num_tasks=5, batch_size=32, epochs=10, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

    wrapper = Blip1Wrapper(
        blip_model=blip_model,
        num_tasks=num_tasks,
        num_embeds=5,
        embed_dim=blip_model.vision_model.config.hidden_size,
        method='third'
    ).to(device)

    # Freeze BLIP backbone and text encoder, only train task embeddings and project layer
    for p in wrapper.vision_model.parameters():
        p.requires_grad = False
    for p in wrapper.text_encoder.parameters():
        p.requires_grad = False
    for emb in wrapper.task_embeddings:
        emb.weight.requires_grad = True
    wrapper.project.requires_grad_(True)

    train_dataset = JsonCaptionDataset(train_json, train_dir, processor.image_processor)
    val_dataset = JsonCaptionDataset(val_json, val_dir, processor.image_processor)
    train_dataset.samples = get_data_balanced(train_dataset.samples)
    val_dataset.samples = get_data_balanced(val_dataset.samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=lambda b: collate_fn(b, processor.tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=lambda b: collate_fn(b, processor.tokenizer))

    optimizer = torch.optim.Adam(wrapper.task_embeddings.parameters(), lr=1e-4)

    best_loss = float("inf")
    best_recall = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        wrapper.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_ids = batch["task_ids"].to(device)

            image_features, text_features = wrapper(pixel_values, input_ids, attention_mask, task_ids)
            loss = compute_loss(image_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        wrapper.eval()
        val_loss = 0.0
        recall_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                task_ids = batch["task_ids"].to(device)

                image_features, text_features = wrapper(pixel_values, input_ids, attention_mask, task_ids)
                loss = compute_loss(image_features, text_features)
                val_loss += loss.item()

                similarity = text_features @ image_features.T
                recall_total += recall_at_k(similarity, k=1)

        avg_val_loss = val_loss / len(val_loader)
        avg_recall = recall_total / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Recall@1: {avg_recall:.4f}")

        if avg_val_loss < best_loss and avg_recall > best_recall:
            print("Saving new best model.")
            best_loss = avg_val_loss
            best_recall = avg_recall
            patience_counter = 0
            torch.save(wrapper.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping.")
                break

def main():
  train_file = sys.argv[1]
  val_file = sys.argv[2]
  train_dir = sys.argv[3]
  val_dir = sys.argv[4]
  save_path = sys.argv[5]
  num_workers = int(sys.argv[6])
  num_tasks = int(sys.argv[7])
  batch_size = int(sys.argv[8])
  epochs = int(sys.argv[9])
  patience = int(sys.argv[10])
  print(train_file, val_file, train_dir, val_dir, save_path, num_workers, num_tasks, batch_size, epochs, patience)
#   train_with_val(train_file, val_file, train_dir,val_dir,save_path,num_workers,num_tasks, batch_size, epochs, patience)
if __name__ == "__main__":
  main()