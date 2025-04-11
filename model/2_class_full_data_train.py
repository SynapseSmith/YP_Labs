import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import math
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torchvision import transforms
from datetime import datetime
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from facenet_pytorch import InceptionResnetV1
import timm
from transformers import ViTForImageClassification, ViTModel
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModel(nn.Module):
    def __init__(self, model_type='vit-base-patch32-384', num_classes=2):
        super(MultiModel, self).__init__()
        self.model_name = model_type
        
        if model_type == 'vit':
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224", 
                ignore_mismatched_sizes=True
            )
            feature_dim = self.model.config.hidden_size

        if model_type == 'vit-base-patch32-384':
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch32-384", 
                ignore_mismatched_sizes=True
            )
            feature_dim = self.model.config.hidden_size

        if model_type == 'vit-large-patch16-224':
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-large-patch16-224", 
                ignore_mismatched_sizes=True
            )
            feature_dim = self.model.config.hidden_size

        if model_type == 'vit-large-patch32-384':
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-large-patch32-384", 
                ignore_mismatched_sizes=True
            )
            feature_dim = self.model.config.hidden_size

        # Custom classifier layers
        self.fc1 = nn.Linear(feature_dim, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        # Additional layers
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        # self.bn1 = nn.LayerNorm(256)
        # self.bn2 = nn.LayerNorm(128)
        # self.bn3 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

        # self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, pixel_values, labels=None):
        if self.model_name.startswith('vit'):
            outputs = self.model(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            x = hidden_states[:, 0, :]  

        else:
            outputs = self.model(pixel_values)
            if hasattr(outputs, 'logits'):
                x = outputs.logits  
            else:
                x = outputs  

        x = self.relu(self.bn1(self.fc1(x)))  
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        # x = self.fc(x)
        
        if labels is not None:
            loss = self.loss_fn(x, labels)
            return loss, x
        else:
            return x

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 3]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_ratio, total_steps, lr_decay, last_epoch=-1):
        def lr_lambda(step):
            warmup_steps = int(total_steps * warmup_ratio)
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return lr_decay ** (step - warmup_steps)
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0).float()
    labels = torch.tensor(labels).long()
    return {"pixel_values": images, "labels": labels}

def main():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터 로드
    train_file = f"C:/Users/honjo/profile_level_prediction/data/5_fold_2class_AC_cleaned_6000/train_fold_1.csv"
    valid_file = f"C:/Users/honjo/profile_level_prediction/data/5_fold_2class_AC_cleaned_6000/valid_fold_1.csv"
   
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    augmentation_count = 1
    train_df = pd.concat([train_df] * augmentation_count, ignore_index=True)
    
    label2id = {'A': 0, 'C': 1}
    id2label = {v: k for k, v in label2id.items()}
    
    print("인덱스 별 레이블 :", label2id)
    
    train_df['cs_rank'] = train_df['cs_rank'].map(label2id)

    # 모델 초기화
    model = MultiModel()
    model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    train_dataset = CustomImageDataset(train_df, transform=train_transform)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = (
        f'runs_{args.num_class}class_full_data_train/{timestamp}_'
        f'model_{model.model_name}_'
        f'ep_{args.epoch}_bs_{args.batch_size}_lr_{args.lr}_ld_{args.lr_decay}_'
        f'wd_{args.weight_decay}_warmup_{args.warmup_ratio}'
    )

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        logging_strategy="epoch",
        eval_strategy="no",
        metric_for_best_model='accuracy',
        # load_best_model_at_end=True,
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        logging_dir=f'{run_dir}/logs',
        seed=1234
    )

    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epoch
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupConstantSchedule(optimizer, warmup_ratio=args.warmup_ratio, total_steps=total_steps, lr_decay=args.lr_decay)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()

    # 모델 저장
    trainer.save_model(run_dir)
    print(f"Model saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=3, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.98, help="Learning rate decay")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio")
    parser.add_argument('--num_class', type=int, default=2, help="Number of classes")
    args = parser.parse_args()
    
    print(args)

    main()