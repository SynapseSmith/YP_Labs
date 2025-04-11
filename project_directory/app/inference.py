import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from facenet_pytorch import MTCNN
from PIL import Image
import torch.nn as nn
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

mtcnn = MTCNN(keep_all=False, device=device)

# MultiModel definition
class MultiModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiModel, self).__init__()

        self.model = ViTForImageClassification.from_pretrained(
            f"google/vit-base-patch32-384", ignore_mismatched_sizes=True
        )
        feature_dim = self.model.config.hidden_size

        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        x = hidden_states[:, 0, :]
        x = self.relu(self.bn1(self.fc1(x)))  
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 1: 3-class prediction
def predict_step1(image, model):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_prob, class_idx = torch.max(probabilities, dim=1)

    labels = ["승인", "기타승인", "비승인"]
    return labels[class_idx.item()], max_prob.item()

# Step 2: 2-class prediction (A, C, B)
def predict_step2(image, model, threshold_a=0.865, threshold_c=0.93):
    cropped_image, _ = detect_and_crop_face(image)
    cropped_tensor = transform(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(cropped_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_prob, class_idx = torch.max(probabilities, dim=1)

    labels = ["A", "C", "B"]
    if class_idx == 0 and max_prob.item() > threshold_a:
        return "A", max_prob.item()
    elif class_idx == 1 and max_prob.item() > threshold_c:
        return "C", max_prob.item()
    else:
        return "B", max_prob.item()

# Face detection and cropping
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
