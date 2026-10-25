import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image
import tqdm
import torch
import torchvision as v2
import fundus_image_toolbox as fit

class EyePacsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, cfg, use_train_transform=False):
        self.csv_file = csv_file
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        self.data_frame = self.data_frame[
            self.data_frame['diagnosis_image_dr_level'].notna() & self.data_frame['image_path'].notna()
        ].reset_index(drop=True)
        self._labels = self.data_frame['diagnosis_image_dr_level']
        self._labels = torch.tensor(self._labels.to_list(), dtype=torch.long)
        self._quality = self.data_frame['session_image_quality']
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data_frame.loc[idx, 'image_path'])
        image = Image.open(image_path)
        image = v2.transforms.ToTensor()(image)
        label = self._labels[idx]
        quality = self._quality[idx]
        return image, label, quality
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = SimpleNamespace(data=SimpleNamespace(image_resolution=1024, csv_file='INSERTYOURPATH', root_dir='INSERTYOURPATH'))

csv_path = cfg.data.csv_file
root_dir = cfg.data.root_dir
ds = EyePacsDataset(csv_path, root_dir=root_dir, cfg=cfg)

ensemble = fit.load_quality_ensemble(device=device)

dataloader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)

confs, labels, qualities = [], [], []
for idx, (images, _, quality) in enumerate(tqdm.tqdm(dataloader)):
    conf, label = fit.ensemble_predict_quality(ensemble, images, threshold=0.25)
    confs.append(conf)
    labels.append(label)
    qualities.append(quality)

confs = np.concatenate(confs, axis=0)
labels = np.concatenate(labels, axis=0)
qualities = np.concatenate(qualities, axis=0)

df = pd.read_csv(csv_path, low_memory=False)
df = df[df['diagnosis_image_dr_level'].notna() & df['image_path'].notna()].reset_index(drop=True)

n = len(confs)
df.loc[:n-1, 'confidence'] = confs
df.loc[:n-1, 'predicted_quality_label'] = labels

df.to_csv('INSERTYOURPATH', index=False)
