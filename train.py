import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import json
import zipfile
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import matplotlib.pyplot as plt
from math import sqrt
import os


class Config:
    
    coco_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    image_base_url = "http://images.cocodataset.org/train2017/"
    image_size = 128
    batch_size = 32
    
    
    text_embed_dim = 768  
    timesteps = 1000      
    
    
    lr = 2e-4
    epochs = 30
    save_every = 5        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnlineCOCO(Dataset):
    def __init__(self):
        self.image_ids = []
        self.captions = []
        self._load_metadata()
        
        self.transform = transforms.Compose([
            transforms.Resize(Config.image_size),
            transforms.CenterCrop(Config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(Config.device)
    
    def _load_metadata(self):
        print("Loading Coco metadata...")
        resp = requests.get(Config.coco_url)
        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            with z.open('annotations/captions_train2017.json') as f:
                data = json.loads(f.read().decode('utf-8'))
        
        self.image_ids = [img['id'] for img in data['images'][:5000]]  # Берем первые 5000 для демо
        self.captions = {ann['image_id']: ann['caption'] for ann in data['annotations'] 
                         if ann['image_id'] in self.image_ids}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_url = f"{Config.image_base_url}{img_id:012d}.jpg"
        
        
        img_resp = requests.get(img_url)
        img = Image.open(BytesIO(img_resp.content)).convert('RGB')
        
        
        caption = self.captions.get(img_id, "")
        inputs = self.tokenizer(caption, return_tensors='pt', padding='max_length', 
                              max_length=64, truncation=True).to(Config.device)
        with torch.no_grad():
            text_emb = self.text_model(**inputs).last_hidden_state.mean(dim=1)
        
        return self.transform(img), text_emb.squeeze()


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        #
        self.time_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Downsample
        self.down1 = self._make_block(3, 64)
        self.down2 = self._make_block(64, 128)
        self.down3 = self._make_block(128, 256)
        
        # Bottleneck
        self.mid = self._make_block(256, 512)
        
        # Upsample
        self.up3 = self._make_block(512 + 256, 128)
        self.up2 = self._make_block(128 + 128, 64)
        self.up1 = self._make_block(64 + 64, 3)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU()
        )
    
    def forward(self, x, t, text_emb):
        
        t = self.time_embed(t.float().view(-1, 1))
        
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        # Bottleneck
        m = self.mid(self.pool(d3))
        
        # Upsample
        u3 = self.up3(torch.cat([self.upsample(m), d3], dim=1))
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1))
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1))
        
        return u1


class Diffusion:
    def __init__(self):
        self.betas = torch.linspace(1e-4, 0.02, Config.timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None].to(x.device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None].to(x.device)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise
    
    def sample(self, model, text_emb, n=4):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, Config.image_size, Config.image_size)).to(Config.device)
            
            for t in tqdm(range(Config.timesteps-1, -1, -1), desc="Генерация"):
                t_batch = torch.full((n,), t, device=Config.device)
                pred_noise = model(x, t_batch, text_emb)
                
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (x - (1 - alpha) * pred_noise / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha)
                x = x + noise * torch.sqrt((1 - alpha) * (1 - alpha_bar) / alpha)
        
        return torch.clamp(x, -1, 1)


def generate_images(model, diffusion, tokenizer, text_model, prompts, n_images=4):
    
    model.eval()
    with torch.no_grad():
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(Config.device)
        text_emb = text_model(**inputs).last_hidden_state.mean(dim=1)
        
        
        generated = diffusion.sample(model, text_emb, n=len(prompts))
        
        
        plt.figure(figsize=(15, 5))
        for i in range(len(prompts)):
            plt.subplot(1, len(prompts), i+1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            plt.title(prompts[i][:30] + "..." if len(prompts[i]) > 30 else prompts[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    
    dataset = OnlineCOCO()
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    model = DiffusionModel().to(Config.device)
    diffusion = Diffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
    
    os.makedirs("saved_models", exist_ok=True)
    
    
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        
        for images, text_embs in tqdm(loader, desc=f"Эпоха {epoch+1}/{Config.epochs}"):
            images = images.to(Config.device)
            text_embs = text_embs.to(Config.device)
            
            
            t = torch.randint(0, Config.timesteps, (images.size(0),), device=Config.device)
            noisy_images, noise = diffusion.add_noise(images, t)
            
            
            pred_noise = model(noisy_images, t, text_embs)
            loss = F.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss/len(loader):.4f}")
        
        
        if (epoch+1) % Config.save_every == 0:
            # Сохраняем модель
            torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch+1}.pth")
            
            
            test_prompts = [
                "a cat sitting on a couch",
                "a dog playing in the park",
                "a sunset over the ocean",
                "a futuristic city at night"
            ]
            generate_images(model, diffusion, dataset.tokenizer, dataset.text_model, test_prompts)

if __name__ == "__main__":
    main()
