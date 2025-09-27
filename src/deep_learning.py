# -*- coding: utf-8 -*-
"""Deep Learning models using DINO and ViT"""

import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, Dinov2Model, CLIPImageProcessor, CLIPVisionModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
general_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

class ViTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls_token = outputs[:, 0]                # [CLS] token
        return self.classifier(cls_token)

class BeeDataset(Dataset):
    def __init__(self, df, image_dir, processor, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.augment = augment
        self.label_map = dict(zip(df["label"], df["bug type"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"preprocessed_{row['ID']}.png")
        mask_path = os.path.join(self.image_dir, f"masks/preprocessed_mask_{row['ID']}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda x: 255 if x > 128 else 0)

        image_np = np.array(image)
        mask_np = np.array(mask)
        masked_image = cv2.bitwise_and(image_np, image_np, mask=mask_np)
        masked_image_pil = Image.fromarray(masked_image)

        if self.augment:
            label_name = self.label_map[row["label"]]
            transform = strong_transform if label_name == "others" else general_transform
            masked_image_tensor = transform(masked_image_pil)
            masked_image_pil = transforms.ToPILImage()(masked_image_tensor)

        inputs = self.processor(images=masked_image_pil, return_tensors="pt")
        inputs["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return inputs

class BeeDatasetViT(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['ID']}.JPG")
        mask_path = os.path.join(self.image_dir, f"masks/binary_{row['ID']}.tif")

        # load and apply mask
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        mask  = mask.point(lambda x: 255 if x > 128 else 0)
        img_np = np.array(image)
        mask_np = np.array(mask)
        masked = cv2.bitwise_and(img_np, img_np, mask=mask_np)
        masked_pil = Image.fromarray(masked)

        inputs = self.processor(images=masked_pil, return_tensors="pt")
        inputs["labels"] = torch.tensor(row["label"])
        return inputs

def load_and_prepare_data():
    df = pd.read_excel("Pollinator-insects-classification/data/classif.xlsx")
    
    bug_type_counts = df['bug type'].value_counts()
    bugs_to_keep = bug_type_counts[bug_type_counts > 1].index
    df = df[df['bug type'].isin(bugs_to_keep)]
    
    df['bug type'] = df['bug type'].replace(['Hover fly', 'Wasp', 'Butterfly'], 'others')
    df["label"] = LabelEncoder().fit_transform(df["bug type"])
    
    print(df["bug type"].unique())
    
    return df

def augment_entire_df(df, label_column="bug type", target_class="others", 
                     n_copies_for_others=3, n_copies_for_non_others=1):
    """
    For every original row in df, emit:
      - 1 "original" copy
      - n_copies_for_others extra copies if bug type == target_class
      - n_copies_for_non_others extra copies if bug type != target_class
    """
    augmented_rows = []
    df["orig_ID"] = df.index
    
    for _, row in df.iterrows():
        # Always keep the original once
        augmented_rows.append(deepcopy(row))

        if row[label_column] == target_class:
            # Add strong augmentation copies for 'others'
            for _ in range(n_copies_for_others):
                augmented_rows.append(deepcopy(row))
        else:
            # Add light augmentation copies for all other classes
            for _ in range(n_copies_for_non_others):
                augmented_rows.append(deepcopy(row))

    return pd.DataFrame(augmented_rows).reset_index(drop=True)

def train_dino_model(df):
    # Create augmented DataFrame
    aug_df = augment_entire_df(df, n_copies_for_others=7, n_copies_for_non_others=2)
    aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("After augmentation, class counts:\n", aug_df["bug type"].value_counts())
    
    # Split data maintaining group structure
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(aug_df, y=aug_df["label"], groups=aug_df["orig_ID"]))
    train_df = aug_df.iloc[train_idx].reset_index(drop=True)
    test_df  = aug_df.iloc[test_idx].reset_index(drop=True)
    
    print("Train size:", len(train_df), "  Test size:", len(test_df))
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained('facebook/webssl-dino3b-full2b-224')
    backbone = Dinov2Model.from_pretrained('facebook/webssl-dino3b-full2b-224')
    
    num_classes = len(df["bug type"].unique())
    model = DinoClassifier(backbone, num_classes=num_classes).to(device)
    
    # Build DataLoaders
    train_dataset = BeeDataset(train_df, "Pollinator-insects-classification/preprocessed", 
                              processor, augment=True)
    test_dataset = BeeDataset(test_df, "Pollinator-insects-classification/preprocessed", 
                             processor, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)
    
    # Setup training
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df["label"]), y=train_df["label"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    num_epochs = 15
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss:.4f}\n")
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report on Test Set:")
    le = LabelEncoder()
    le.fit(df["bug type"])
    print(classification_report(all_labels, all_preds, target_names=le.inverse_transform(np.unique(all_labels))))
    
    return model

def train_vit_model(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=0)

    processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", 
                                                  do_resize=True, size=224, do_normalize=True)
    backbone = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    model = ViTClassifier(backbone).to(device)

    train_dataset = BeeDatasetViT(train_df, "Pollinator-insects-classification/data", processor)
    test_dataset = BeeDatasetViT(test_df, "Pollinator-insects-classification/data", processor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=12)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}\n")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].squeeze(1).to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    
    return model