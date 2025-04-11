import model.demo_inference as gr
import torch
import argparse
from torchvision import transforms
from transformers import ViTForImageClassification, ViTModel
from PIL import Image, ImageDraw
from facenet_pytorch import InceptionResnetV1, MTCNN
import timm
import torch.nn as nn
from safetensors.torch import load_file  
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 얼굴 검출 및 크롭 함수 정의
def detect_and_crop_face(image):
    # 얼굴 검출
    boxes, _ = mtcnn.detect(image)
    if boxes is not None and len(boxes) > 0:
        # 가장 큰 얼굴 선택
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_box = boxes[np.argmax(areas)]
        
        # 좌표 추출 및 이미지 크롭
        x0, y0, x1, y1 = [int(coord) for coord in largest_box]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.width, x1)
        y1 = min(image.height, y1)
        
        cropped_image = image.crop((x0, y0, x1, y1))
        
        # 바운딩 박스 그리기
        draw = ImageDraw.Draw(image)
        draw.rectangle([x0, y0, x1, y1], outline="green", width=3)

        return cropped_image, image  # 크롭된 이미지와 바운딩 박스를 포함한 이미지를 반환
    else:
        return image, image  # 얼굴을 찾지 못하면 원본 이미지를 반환

# 추론을 위한 함수 정의
def predict(image, model1_path, model2_path, model1_type='vit-large-patch32-384', model2_type='vit-large-patch32-384', num_classes1=3, num_classes2=2, threshold=0.7):
    # 얼굴 검출 및 크롭 먼저 수행
    cropped_image, image_with_box = detect_and_crop_face(image)
    
    # 크롭된 이미지로 3-class 모델 추론 수행
    cropped_image_tensor = transform(cropped_image).unsqueeze(0).to(device)

    # 1차 모델 로딩 및 추론 (3-class 분류기)
    model1 = MultiModel(model_type=model1_type, num_classes=num_classes1)
    model1_weights = load_file(model1_path)  # safetensors 파일 로드
    model1.load_state_dict(model1_weights)
    model1.to(device)
    model1.eval()

    with torch.no_grad():
        outputs1 = model1(cropped_image_tensor)
        predictions1 = torch.argmax(outputs1, dim=1).cpu().numpy()
    
    # ID2Label 맵핑
    id2label1 = {0: '일반', 1: '기타승인', 2: '비승인'}
    primary_label = id2label1[predictions1[0]]

    # 2차 모델로 분류기 적용 (2-class 분류기), '일반' 경우에만 추가 분류
    if primary_label == '일반':
        model2 = MultiModel(model_type=model2_type, num_classes=num_classes2)
        model2_weights = load_file(model2_path)  # safetensors 파일 로드
        model2.load_state_dict(model2_weights)
        model2.to(device)
        model2.eval()

        with torch.no_grad():
            outputs2 = model2(cropped_image_tensor)
            max_prob = torch.max(torch.softmax(outputs2, dim=1)).item()
            predictions2 = torch.argmax(outputs2, dim=1).cpu().numpy()

        id2label2 = {0: 'A', 1: 'C'}
        
        # 최대 출력값이 임계값 이하라면 'B'로 분류
        if max_prob <= threshold:
            final_label = 'B'
        else:
            final_label = id2label2[predictions2[0]]

        result = f"2-class (A, C, B): {final_label}\nProbability: {max_prob:.2f}"
        if final_label == 'B':
            result = f"2-class (A, C, B): {final_label}\nProbability: {max_prob:.2f} from {id2label2[predictions2[0]]}"
    else:
        cropped_image = image  # 얼굴이 감지되지 않았을 때, 원본 이미지 사용
        result = f"3-class (일반, 기타승인, 비승인): {primary_label}"

    return result, image_with_box, cropped_image

def gradio_interface(model1_path, model2_path, model1_type='vit-large-patch32-384', model2_type='vit-large-patch32-384', num_classes1=3, num_classes2=2, threshold_a=0.9, threshold_c=0.9):
    def predict_image(image):
        result, image_with_box  = predict(image, model1_path, model2_path, model1_type, model2_type, threshold_a, threshold_c)
        # HTML 형식으로 텍스트 반환
        result_html = f'<div style="font-size: 36px; font-weight: bold; color: black;">{result}</div>'
        return result_html, image_with_box

    gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=[gr.HTML(), gr.Image(type="pil")],  # HTML 출력으로 변경
        examples=[# A 클래스
                  ["./data/images/2022-07-18/0d53a172-1dd7-4e6f-b51d-8e2284a07324.jpeg"], 
                  ["./data/images/2021-02-13/977963cf-3aac-4ea7-8655-71d889418c4c.jpeg"],
                  ["./data/images/2021-05-11/e69f4ac4-05ed-480f-8858-646b00927bf0.jpeg"],
                  # B 클래스
                  ["./data/images/2024-01-16/0e9924ad-927e-4b4c-9c6a-0bbfa01e5305.jpeg"],
                  ["./data/images/2024-01-09/0f4f9867-3063-4346-9b18-141bd78e1216.jpeg"],
                  ["./data/images/2024-01-16/6c84638a-80eb-4c11-bf4d-f7f4bbd45460.jpeg"],
                  # C 클래스
                  ["./data/images/2022-10-15/ddd44f81-2046-4415-9548-07402ff8900e.jpeg"],
                  ["./data/images/2024-03-27/0e3fab05-40cf-4db8-8cee-f5981ff2cb71.jpeg"],
                  ["./data/images/2021-01-09/a133c262-2a16-45be-8e8f-8d1026804aca.jpeg"],
                  # 기타 승인
                  ["./data/images/2024-06-10/ff644731-e949-47f3-89d8-9b0c066da185.jpeg"],
                  ["./data/images/2020-06-10/d41086a4-528b-4e73-a458-978c690fe373.jpeg"],
                  ["./data/images/2022-07-27/60e32290-17a7-4522-9ee5-dc287ba49651.jpeg"],
                  # 비승인
                  ["C:/Users/honjo/profile_level_prediction/data/qualitative_analysis/true_positives_nsfw/true_positive_1437.png"],
                  ["C:/Users/honjo/profile_level_prediction/data/qualitative_analysis/true_positives_nsfw/true_positive_1723.png"],
                  ["C:/Users/honjo/profile_level_prediction/data/qualitative_analysis/true_positives_nsfw/true_positive_1779.png"]
                  ]
    ).launch(debug=True, share=True)

# 실제 실행 시
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_path', type=str, default="runs_3class_normal_null_nsfw/20241022_173851_model_vit-large-patch32-384_ep_100_bs_32_lr_5e-05_ld_0.9997_wd_1e-05_warmup_0.1_fold_1/checkpoint-2584/model.safetensors", help="Path to the trained model 1 checkpoint")
    parser.add_argument('--model2_path', type=str, default="runs_2class_5fold_face_cropped_FBP_AC/20241025_174422_model_vit-large-patch32-384_ep_50_bs_64_lr_5e-05_ld_0.997_wd_1e-05_warmup_0.1_fold_1/checkpoint-663/model.safetensors", help="Path to the trained model 2 checkpoint")
    parser.add_argument('--model1_type', type=str, default='vit-large-patch32-384', help="Model 1 type")
    parser.add_argument('--model2_type', type=str, default='vit-large-patch32-384', help="Model 2 type")
    parser.add_argument('--num_classes1', type=int, default=3, help="Number of classes for model 1")
    parser.add_argument('--num_classes2', type=int, default=2, help="Number of classes for model 2")
    parser.add_argument('--threshold', type=float, default=0.9, help="Probability threshold for classification")
    args = parser.parse_args()

    gradio_interface(args.model1_path, args.model2_path, args.model1_type, args.model2_type, args.num_classes1, args.num_classes2, args.threshold)
