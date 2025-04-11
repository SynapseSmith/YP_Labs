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
from PIL import Image, ImageFile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModel(nn.Module):
    def __init__(self, model_type='vit-base-patch32-384'):
        super(MultiModel, self).__init__()
        self.model_name = model_type

        # Initialize model based on model_type
        if model_type == 'inceptionresnetv1':
            self.model = InceptionResnetV1(pretrained='vggface2')
            feature_dim = 512

        if model_type == 'inceptionresnetv2':
            self.model = timm.create_model('inception_resnet_v2', pretrained=True)
            feature_dim = self.model.classif.in_features 
            self.model.classif = nn.Identity()  # Remove the classifier to add custom layers

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
        
        # 추가된 모델들
        if model_type == 'vit-huge-patch14-224':
            # ViT Huge (~632M parameters)
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-huge-patch14-224-in21k", 
                ignore_mismatched_sizes=True
            )
            feature_dim = self.model.config.hidden_size  # Hidden size for ViT-huge is 1280

        if model_type == 'swinv2-giant-patch4-window12-384':
            # SwinV2 Giant (~3B parameters)
            self.model = timm.create_model('swinv2_giant_patch4_window12_384', pretrained=True)
            feature_dim = self.model.head.in_features  # Hidden size for SwinV2 Giant is 2048
            self.model.head = nn.Identity()  # Remove the classifier

        if model_type == 'convnext-xxlarge-384':
            # ConvNeXt XXLarge (~3.5B parameters)
            self.model = timm.create_model('convnext_xxlarge', pretrained=True)
            feature_dim = self.model.head.fc.in_features  # Hidden size for ConvNeXt-XXLarge is 2048
            self.model.head = nn.Identity()  # Remove the classifier

        if model_type == 'beit-large-patch16-224':
            # Beit Large (~307M parameters)
            self.model = timm.create_model('beit_large_patch16_224', pretrained=True)
            feature_dim = self.model.head.in_features  # Hidden size for Beit-Large is 1024
            self.model.head = nn.Identity()  # Remove the classifier

        # Custom classifier layers
        self.fc1 = nn.Linear(feature_dim, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, args.num_class)

        # Additional layers
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        if self.model_name.startswith('vit'):
            outputs = self.model(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Extract the last hidden state
            x = hidden_states[:, 0, :]  # CLS token representation

        else:
            outputs = self.model(pixel_values)
            if hasattr(outputs, 'logits'):
                x = outputs.logits  # Extract the logits for classification models
            else:
                x = outputs  # Use the direct output for other models
            
        # Pass through the custom classifier layers
        x = self.relu(self.bn1(self.fc1(x)))  
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)

        if labels is not None:
            loss = self.loss_fn(x, labels)
            return loss, x
        else:
            return x

# 손상된 이미지 파일을 무시하도록 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    # __getitem__ 수정
    def __getitem__(self, idx):
        img_data = self.dataframe.iloc[idx, 0]
        
        try:
            # 이미지가 파일 경로인 경우와 PIL 객체인 경우를 구분
            if isinstance(img_data, str):  # 파일 경로인 경우
                image = Image.open(img_data).convert('RGB')
            elif isinstance(img_data, Image.Image):  # PIL 객체인 경우
                image = img_data
            else:
                raise ValueError(f"Unsupported image data type: {type(img_data)}")

            label = self.dataframe.iloc[idx, 1]
            
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {img_data}: {e}")
            # 대체용 빈 이미지 또는 특정 조치를 취할 수 있습니다.
            # 여기서는 0으로 채워진 이미지(검은색 이미지)로 대체합니다.
            image = Image.new('RGB', (384, 384), (0, 0, 0))
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
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        print(f"Predictions shape: {preds.shape}, Labels shape: {labels.shape}")
        # print(f"Predictions: {preds}, Labels: {labels}")
        if preds.shape[0] != labels.shape[0]:
            min_len = min(preds.shape[0], labels.shape[0])
            preds = preds[:min_len]  # 크기 일치시키기 위해 자르기
            labels = labels[:min_len]
            
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,  # Macro F1
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted  # Weighted F1
        }

        for label_id, label_name in id2label.items():
            # 각 클래스에 대한 TP 계산 (예측과 실제 값이 일치하는 경우)
            true_positive = ((preds == labels) & (labels == label_id)).sum()
            # 해당 클래스에 속하는 전체 샘플 수
            total_samples = (labels == label_id).sum()
            class_accuracy = true_positive / total_samples if total_samples > 0 else 0
            # 클래스별 accuracy 추가
            metrics[f"accuracy_{label_name}"] = class_accuracy
            metrics[f"precision_{label_name}"] = precision[label_id]
            metrics[f"recall_{label_name}"] = recall[label_id]
            metrics[f"f1_{label_name}"] = f1[label_id]

        return metrics
    
    # 5-fold cross-validation
    for fold in range(1, 6):
        print(f"Fold {fold}")
        
        train_file = f"C:/Users/honjo/profile_level_prediction/data/5_fold_3n_cleaned_null_maximization/train_fold_{fold}.csv"
        valid_file = f"C:/Users/honjo/profile_level_prediction/data/5_fold_3n_cleaned_null_maximization/valid_fold_{fold}.csv"
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(valid_file)
            
        label2id = {'normal': 0, '-': 1, 'nsfw': 2}
        id2label = {v: k for k, v in label2id.items()}
    
        print("인덱스 별 레이블 :", label2id)
        
        train_df['cs_rank'] = train_df['cs_rank'].map(label2id)
        val_df['cs_rank'] = val_df['cs_rank'].map(label2id)
        
        print(train_df['cs_rank'].unique())
        print(val_df['cs_rank'].unique())
        
        model = MultiModel()
        model.to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {params:,}")
        
        train_dataset = CustomImageDataset(train_df, transform=train_transform)
        val_dataset = CustomImageDataset(val_df, transform=eval_transform)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        

        run_dir = (
            f'runs_{args.num_class}class_5fold_null_maximization/{timestamp}_'
            f'model_{model.model_name}_'
            f'ep_{args.epoch}_bs_{args.batch_size}_lr_{args.lr}_ld_{args.lr_decay}_'
            f'wd_{args.weight_decay}_warmup_{args.warmup_ratio}_fold_{fold}'
        )

        training_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=args.epoch,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model='accuracy',
            load_best_model_at_end=True,
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
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            args=training_args,
            data_collator=collate_fn,
            optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()

        metrics = trainer.evaluate()
        print(f"Final evaluation metrics for fold {fold}: {metrics}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.9998, help="Learning rate decay")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="Weight decay for optimizer")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio")
    parser.add_argument('--num_class', type=int, default=3, help="Number of classes")
    args = parser.parse_args()
    
    print(args)

    main()
