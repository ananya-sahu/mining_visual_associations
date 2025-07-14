import os
import json
import random
import sys
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F

# --- Balance Samples ---
def get_data_balanced(samples):
    """
    Reorganize samples so that labels are evenly distributed per image.
    Each sample is a tuple: (img_path, caption, task_label)
    """
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
                full_img_path = os.path.join(home_dir, filename)
                self.samples.append((full_img_path, caption, int(task_label) - 1))  # zero-index labels

        self.image_processor = image_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, task_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return image, caption, torch.tensor(task_label, dtype=torch.long)

# --- Collate Function ---
def collate_fn(batch, tokenizer, max_length=64):
    images, captions, task_labels = zip(*batch)
    images = torch.stack(images)
    tokenized = tokenizer(
        list(captions),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    return {
        "pixel_values": images,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "task_ids": torch.stack(task_labels)
    }

# --- SigLIP Wrapper ---
class SigLIPWrapper(nn.Module):
    def __init__(self, siglip_model, num_tasks, num_embeds, embed_dim, method='third'):
        super().__init__()
        self.siglip_model = siglip_model
        self.task_embeddings = nn.ModuleList([
            nn.Embedding(num_tasks, embed_dim) for _ in range(num_embeds)
        ])
        self.method = method

    def encode_image(self, pixel_values, task_ids):
        vision_outputs = self.siglip_model.vision_model(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state  # (B, num_patches+1, D)

        task_embeds = torch.stack([embed(task_ids) for embed in self.task_embeddings], dim=1)

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
        text_outputs = self.siglip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = text_outputs.last_hidden_state[:, 0, :]
        return F.normalize(pooled, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask, task_ids):
        image_features = self.encode_image(pixel_values, task_ids)
        text_features = self.encode_text(input_ids, attention_mask)
        return image_features, text_features

# --- Loss ---
def compute_loss(image_features, text_features):
    sim_i2t = image_features @ text_features.T
    sim_t2i = text_features @ image_features.T
    labels = torch.eye(sim_i2t.size(0), device=sim_i2t.device)
    loss_i2t = F.binary_cross_entropy_with_logits(sim_i2t, labels)
    loss_t2i = F.binary_cross_entropy_with_logits(sim_t2i, labels)
    return (loss_i2t + loss_t2i) / 2

# --- Recall@K ---
def recall_at_k(similarity_matrix, k):
    recalls = 0
    for i in range(similarity_matrix.shape[0]):
        top_k = torch.topk(similarity_matrix[i], k=k).indices
        if i in top_k:
            recalls += 1
    return recalls / similarity_matrix.shape[0]

# --- Training with Val ---
def train_with_val(train_json, val_json, train_dir, val_dir, save_path,num_workers=8, num_tasks=5, batch_size=64, epochs=10, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    wrapper = SigLIPWrapper(
        model, num_tasks=num_tasks,
        num_embeds=5,
        embed_dim=model.config.vision_config.hidden_size
    ).to(device)

    for param in wrapper.siglip_model.parameters():
        param.requires_grad = False
    for param in wrapper.task_embeddings.parameters():
        param.requires_grad = True

    # Load datasets
    train_dataset = JsonCaptionDataset(train_json, train_dir, processor.image_processor)
    val_dataset = JsonCaptionDataset(val_json, val_dir, processor.image_processor)

    # Balance datasets
    train_dataset.samples = get_data_balanced(train_dataset.samples)
    val_dataset.samples = get_data_balanced(val_dataset.samples)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer)
    )

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

        # Validation
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

        # Save best model
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
#   train_with_val(train_file, val_file, train_dir,val_dir,save_path,num_workers,num_tasks, batch_size, epochs, patience)
  print(train_file, val_file, train_dir, val_dir, save_path, num_workers, num_tasks, batch_size, epochs, patience)
if __name__ == "__main__":
  main()