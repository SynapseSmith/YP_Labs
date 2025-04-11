import os
import time
import requests
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
from safetensors.torch import load_file
from torchvision.transforms.functional import to_pil_image
from inference import MultiModel, predict_step1, predict_step2, device, transform

app = FastAPI()

step1_model_path = "3n_class_classifier/model.safetensors"  # 3-class 모델
male_step2_model_path = "male_classifier/model.safetensors"  # 남성용 2-class 모델
female_step2_model_path = "female_clasifier/model.safetensors"  # 여성용 2-class 모델

step1_model = MultiModel(num_classes=3)
step1_weights = load_file(step1_model_path)
step1_model.load_state_dict(step1_weights)
step1_model.to(device).eval()

male_step2_model = MultiModel(num_classes=2)
male_step2_weights = load_file(male_step2_model_path)
male_step2_model.load_state_dict(male_step2_weights)
male_step2_model.to(device).eval()

female_step2_model = MultiModel(num_classes=2)
female_step2_weights = load_file(female_step2_model_path)
female_step2_model.load_state_dict(female_step2_weights)
female_step2_model.to(device).eval()

@app.post("/predict/")
async def classify_images(image_paths: List[str], gender: str, threshold_a: Optional[float] = None, threshold_c: Optional[float] = None):
    """
    Handle batch image classification requests with gender-specific models.
    """
    if gender.lower().strip() not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Invalid gender. Choose 'male' or 'female'.")

    results = []
    for image_path in image_paths:
        start_time = time.time()
        try:
            # Check if the input is a URL or local path
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # Fetch image from URL
                response = requests.get(image_path)
                if response.status_code != 200:
                    raise HTTPException(status_code=404, detail=f"Unable to fetch image from URL: {image_path}")
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Check if file exists locally
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"File not found: {image_path}")
                img = Image.open(image_path).convert("RGB")

            # Step 1: 3-class prediction
            step1_label, step1_confidence = predict_step1(img, step1_model)

            if step1_label == "기타승인":
                result = {
                    "filename": image_path,
                    "approval_label": 'pending_approval',
                    "score": step1_confidence,
                    "gender": gender,
                    "processing_time": f"{time.time() - start_time:.2f} seconds",
                    "status": "success",
                    "message": "Image classified as pending-approval."
                }
            elif step1_label == "비승인":
                result = {
                    "filename": image_path,
                    "approval_label": 'disapproved',
                    "score": step1_confidence,
                    "gender": gender,
                    "processing_time": f"{time.time() - start_time:.2f} seconds",
                    "status": "success",
                    "message": "Image classified as non-approvable."
                }
            else:
                step2_model = male_step2_model if gender == "male" else female_step2_model

                if gender == 'male':
                    threshold_a = threshold_a if threshold_a is not None else 0.865
                    threshold_c = threshold_c if threshold_c is not None else 0.93
                elif gender == 'female':
                    threshold_a = threshold_a if threshold_a is not None else 0.92
                    threshold_c = threshold_c if threshold_c is not None else 0.951

                step2_label, step2_confidence = predict_step2(img, step2_model, threshold_a, threshold_c)

                result = {
                    "filename": image_path,
                    "approval_label": 'approved',
                    "grade_label": step2_label,
                    "score": step2_confidence,
                    "gender": gender,
                    "processing_time": f"{time.time() - start_time:.2f} seconds",
                    "status": "success",
                    "message": "Profile classification completed."
                }

            results.append(result)

        except Exception as e:
            results.append({
                "filename": image_path,
                "status": "error",
                "message": str(e),
                "processing_time": f"{time.time() - start_time:.2f} seconds",
            })

    return JSONResponse(content=results)