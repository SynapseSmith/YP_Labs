import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torch.nn as nn
from collections import Counter
from safetensors.torch import load_file
import numpy as np
from matplotlib import font_manager, rc
from tqdm import tqdm
import pandas as pd
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# MTCNN face detector 생성
mtcnn = MTCNN(keep_all=False, device=device)

class MultiModel(nn.Module):
    def __init__(self, model_type='vit-large-patch32-384', num_classes=3):
        super(MultiModel, self).__init__()
        self.model_name = model_type
        
        if model_type == 'vit-base-patch32-384':
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch32-384", 
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
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()
        
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
        
        if labels is not None:
            loss = self.loss_fn(x, labels)
            return loss, x
        else:
            return x

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 얼굴 검출 및 크롭 함수 정의
def detect_and_crop_face(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_box = boxes[np.argmax(areas)]
        x0, y0, x1, y1 = [int(coord) for coord in largest_box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        return image.crop((x0, y0, x1, y1)), image
    return image, image

# Temperature scaling 적용
def apply_temperature(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=1)

# 예측 분포를 비교하는 그래프
# 예측 분포를 비교하는 그래프
def plot_comparison(predictions):
    pred_count = Counter(predictions)

    ordered_labels = ['A', 'B', 'C', '기타승인', '비승인']  
    pred_values = [pred_count[label] if label in pred_count else 0 for label in ordered_labels]

    width = 0.35  # 막대 너비
    x = np.arange(len(ordered_labels))  # 막대의 x 위치 계산

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, pred_values, width, label='예측 샘플 수', color='lightgreen')

    ax.set_ylabel('개수', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels, fontsize=20)
    ax.legend(fontsize=16)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, int(yval), ha='center', va='bottom', fontsize=18)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    plt.yticks(fontsize=16)
    plt.ylim(0, max(pred_values) + 300)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# 수정된 predict 함수
def predict(folder_or_csv_path, model1_path, model2_path, model1_type='vit-base-patch32-384',
            model2_type='vit-base-patch32-384', num_classes1=3, num_classes2=2, threshold_a=0.9, threshold_c=0.9):
    if os.path.isdir(folder_or_csv_path):
        image_paths = [os.path.join(folder_or_csv_path, img) for img in os.listdir(folder_or_csv_path)]
        actual_labels = None
    elif folder_or_csv_path.endswith('.csv'):
        df = pd.read_csv(folder_or_csv_path)
        df = df.sample(10000, random_state=42)
        image_paths = df['local_image'].tolist()
        # actual_labels = df['cs_rank'].replace('-', '기타승인').tolist()
    else:
        raise ValueError("Input path must be a folder or a CSV file.")

    model1 = MultiModel(model_type=model1_type, num_classes=num_classes1)
    model1.load_state_dict(load_file(model1_path))
    model1.to(device).eval()

    model2 = MultiModel(model_type=model2_type, num_classes=num_classes2)
    model2.load_state_dict(load_file(model2_path))
    model2.to(device).eval()

    predictions = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs1 = model1(image_tensor)
                predictions1 = torch.argmax(outputs1, dim=1).cpu().numpy()

            if predictions1[0] == 0:
                cropped_image, _ = detect_and_crop_face(image)
                cropped_tensor = transform(cropped_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs2 = model2(cropped_tensor)
                    probabilities = apply_temperature(outputs2, temperature=1.0)
                    max_prob = torch.max(probabilities).item()
                    predictions2 = torch.argmax(probabilities, dim=1).cpu().numpy()

                if predictions2[0] == 0 and max_prob > threshold_a:
                    final_label = 'A'
                elif predictions2[0] == 1 and max_prob > threshold_c:
                    final_label = 'C'
                else:
                    final_label = 'B'
            else:
                final_label = '기타승인' if predictions1[0] == 1 else '비승인'

            predictions.append(final_label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return predictions

# 메인 실행
if __name__ == "__main__":
    csv_path = "data/male.csv"
    model1_path = r"runs_3class_5fold_train\20241204_170045_model_vit-base-patch32-384_ep_50_bs_32_lr_5e-05_ld_0.9997_wd_1e-05_warmup_0.1_fold_1\checkpoint-7104\model.safetensors"
    model2_path = r"runs_2class_5fold_male\20241205_194304_model_vit-base-patch32-384_ep_20_bs_32_lr_5e-05_ld_0.997_wd_1e-05_warmup_0.1_fold_2\checkpoint-666\model.safetensors"
    threshold_a = 0.87
    threshold_c = 0.9325

    predictions = predict(csv_path, model1_path, model2_path,
                                         model1_type='vit-base-patch32-384',
                                         model2_type='vit-base-patch32-384',
                                         num_classes1=3, num_classes2=2,
                                         threshold_a=threshold_a, threshold_c=threshold_c)
    
    plot_comparison(predictions)
