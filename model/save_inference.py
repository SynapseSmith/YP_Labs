import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ViTForImageClassification, ViTModel
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1, MTCNN
import timm
import torch.nn as nn
from collections import Counter
from safetensors.torch import load_file
import numpy as np
from matplotlib import font_manager, rc
from tqdm import tqdm
import pandas as pd
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print('device:', device)


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

def draw_prediction(image, text, font_path="C:/Windows/Fonts/malgun.ttf"):
    # 이미지 크기 기반으로 폰트 크기 동적 설정
    image_width, image_height = image.size
    font_size = max(15, int(image_width * 0.03))  # 너비의 3%를 폰트 크기로, 최소 15
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트의 위치 계산
    text_position = (int(image_width * 0.02), int(image_height * 0.02))  # 이미지의 2% 지점에 텍스트 위치

    # 텍스트 배경 생성
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox(text_position, text, font=font)  # 텍스트 바운딩 박스 계산
    background_position = (text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5)
    draw.rectangle(background_position, fill=(0, 0, 0))  # 검은 배경

    # 굵은 텍스트 그리기 (약간씩 다른 위치에 반복 그리기)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상하좌우로 약간씩 이동
    for offset in offsets:
        draw.text(
            (text_position[0] + offset[0], text_position[1] + offset[1]),
            text,
            fill="green",
            font=font
        )
    # 가운데 텍스트 한 번 더 그리기
    draw.text(text_position, text, fill="green", font=font)

    return image

def apply_temperature(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=1)

# 수정된 predict 함수
def predict(image_paths, model1_path, model2_path, model1_type='vit-base-patch32-384',
            model2_type='vit-base-patch32-384', num_classes1=3, num_classes2=2, threshold_a=0.9, threshold_c=0.9):
    if isinstance(image_paths, str):
        if os.path.isdir(image_paths):
            image_paths = [os.path.join(image_paths, img) for img in os.listdir(image_paths)]
            actual_labels = None
        elif image_paths.endswith('.csv'):
            df = pd.read_csv(image_paths)
            image_paths = df['local_image'].tolist()
            actual_labels = df['profile_level'].replace('-', '기타승인').tolist()
        else:
            raise ValueError("Input path must be a folder or a CSV file.")
    else:
        actual_labels = None

    model1 = MultiModel(model_type=model1_type, num_classes=num_classes1)
    model1.load_state_dict(load_file(model1_path))
    model1.to(device).eval()

    model2 = MultiModel(model_type=model2_type, num_classes=num_classes2)
    model2.load_state_dict(load_file(model2_path))
    model2.to(device).eval()

    predictions = []
    # checkpoint = model2_path.split('//')[1][-1]
    # print(checkpoint)
    
    #sex = image_paths[0].split('//')[1].split('.')[0]
    #output_folder = f"inference_images/threshold_a_{threshold_a}_threshold_c_{threshold_c}"
    # output_folder = f'inference_images/nlp_data'
    os.makedirs(output_folder, exist_ok=True)
    class_folders = {label: os.path.join(output_folder, label) for label in ['A', 'B', 'C', '기타승인', '비승인']}
    for folder in class_folders.values():
        os.makedirs(folder, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs1 = model1(image_tensor)
                softmax1 = torch.softmax(outputs1, dim=1).cpu().numpy()
                predictions1 = np.argmax(softmax1, axis=1)

            if predictions1[0] == 0:
                cropped_image, _ = detect_and_crop_face(image)
                cropped_tensor = transform(cropped_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs2 = model2(cropped_tensor)
                    
                    # Temperature 적용 (수정된 부분)
                    probabilities = apply_temperature(outputs2, temperature=1)
                    
                    max_prob = torch.max(probabilities).item()
                    
                    predictions2 = torch.argmax(probabilities, dim=1).cpu().numpy()
                   
                    
                # 각각의 threshold에 따라 결과 결정
                if predictions2[0] == 0 and max_prob > threshold_a:
                    final_label = 'A'
                    final_prob = f'{probabilities[0][predictions2[0]]:.3f}'
                elif predictions2[0] == 1 and max_prob > threshold_c:
                    final_label = 'C'
                    final_prob = f'{probabilities[0][predictions2[0]]:.3f}'
                else:
                    final_label = 'B'
                    final_prob = f'{max_prob:.3f}'
            
                print(f'probability: {max_prob:.3f}, final_label: {final_label}')        
                
            else:
                final_label = '기타승인' if predictions1[0] == 1 else '비승인'
                final_prob = f'{softmax1[0][0]:.3f}, {softmax1[0][1]:.3f}, {softmax1[0][2]:.3f}'

            predictions.append(final_label)
            text = f"Class: {final_label}, Prob: {final_prob}"
            #image_with_text = draw_prediction(image, text)
            image.save(os.path.join(class_folders[final_label], os.path.basename(img_path)))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return predictions, actual_labels

# 검증 셋 생성 함수
def create_validation_sets(data, num_sets=1, set_size=3000):
    return [data.sample(n=set_size, random_state=123).reset_index(drop=True) for _ in range(num_sets)]

# 검증 셋 테스트 함수
def test_on_validation_sets(csv_path, model1_path, model2_path, model1_type, model2_type, num_classes1, num_classes2, threshold_a, threshold_c):
    validation_sets = create_validation_sets(pd.read_csv(csv_path), num_sets=1, set_size=len(pd.read_csv(csv_path)))
    for idx, val_set in enumerate(validation_sets):
        print(f"## Testing on validation set {idx + 1}")
        image_paths = val_set['local_image'].tolist()
        predict(image_paths, model1_path, model2_path, model1_type, model2_type, num_classes1, num_classes2, threshold_a, threshold_c)

# 메인 실행
if __name__ == "__main__":
    csv_path = "data/female.csv"
    model1_path = r"runs_3class_5fold_null_maximization\20241218_134845_model_vit-base-patch32-384_ep_50_bs_32_lr_5e-05_ld_0.9998_wd_1e-05_warmup_0.1_fold_1\checkpoint-9707\model.safetensors"
    model2_path = r"runs_2class_train_valid_test\20250103_154546_model_vit-large-patch32-384_ep_10_bs_32_lr_5e-05_ld_0.9993_wd_1e-05_warmup_0.1\checkpoint-750\model.safetensors"
    
    epoch = model2_path.split('ep')[1].split('_')[1]
    bs = model2_path.split('bs')[1].split('_')[1]
    wd = model2_path.split('wd')[1].split('_')[1]
    
    threshold_a = 0.935
    threshold_c = 0.945
    # output_folder = f'inference_images/nlp_epoch_{epoch}_bs_{bs}_wd_{wd}_{threshold_a}_{threshold_c}'
    # output_folder = f'inference_images/female'
    output_folder = f'inference_images/null_maximizatdion_female_vit-large'
    
    start_time = time()
    test_on_validation_sets(csv_path, model1_path, model2_path, 
                            model1_type='vit-base-patch32-384', 
                            model2_type='vit-large-patch32-384', 
                            num_classes1=3, num_classes2=2, threshold_a=threshold_a, threshold_c=threshold_c)
    print(f"Total time: {time() - start_time:.2f} seconds")
