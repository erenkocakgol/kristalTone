import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

class MIDIDataset(Dataset):
    def __init__(self, json_folder_path, transform=None, validation_split=0.2):
        self.data = []
        self.transform = transform
        for file_name in os.listdir(json_folder_path):
            if file_name.endswith('.json'):
                with open(os.path.join(json_folder_path, file_name), 'r') as json_file:
                    data = json.load(json_file)
                    self.data.extend(data)

        self.df = pd.DataFrame(self.data)

        # Veri setini karıştır ve eğitim/doğrulama için ayır
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        split_idx = int(len(self.df) * (1 - validation_split))
        self.train_df = self.df[:split_idx]
        self.val_df = self.df[split_idx:]

        # Eğitim veya doğrulama setini seçmek için bayrak
        self.is_val = False

    def use_validation(self):
        self.is_val = True

    def use_training(self):
        self.is_val = False

    def __len__(self):
        if self.is_val:
            return len(self.val_df)
        else:
            return len(self.train_df)

    def __getitem__(self, idx):
        if self.is_val:
            item = self.val_df.iloc[idx]
        else:
            item = self.train_df.iloc[idx]

        sample = {'start': item['start'], 'end': item['end'], 'pitch': item['pitch'], 'velocity': item['velocity']}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

def to_tensor(sample):
    start, end, pitch, velocity = sample['start'], sample['end'], sample['pitch'], sample['velocity']
    return torch.tensor([start, end, pitch, velocity], dtype=torch.float)

# Örnek kullanım
# json_folder_path = "path/to/json/folder"
# dataset = MIDIDataset(json_folder_path=json_folder_path, transform=to_tensor)
# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# dataset.use_validation()  # Doğrulama setini kullan
# val_loader = DataLoader(dataset, batch_size=4, shuffle=False)