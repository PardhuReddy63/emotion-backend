from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from deepface import DeepFace
import base64
import uuid
import os

app = FastAPI(title="Emotion Detection API")

# -----------------------------
# LOAD TEXT MODEL
# -----------------------------
print("Loading Text Emotion Model...")
text_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)
print("Text Model Loaded Successfully!")


# -----------------------------
# REQUEST MODELS
# -----------------------------
class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image: str  # base64 string


# -----------------------------
# TEXT ENDPOINT
# -----------------------------
@app.post("/predict-text")
def predict_text(request: TextRequest):
    try:
        result = text_classifier(request.text)
        emotion = result[0]["label"]
        confidence = round(result[0]["score"] * 100, 2)

        return {
            "emotion": emotion.capitalize(),
            "confidence": f"{confidence}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# IMAGE ENDPOINT (BASE64)
# -----------------------------
@app.post("/predict-image")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)

        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(image_data)

        result = DeepFace.analyze(
            img_path=temp_filename,
            actions=["emotion"],
            enforce_detection=False
        )

        dominant_emotion = result[0]["dominant_emotion"]
        confidence = round(result[0]["emotion"][dominant_emotion], 2)

        os.remove(temp_filename)

        return {
            "emotion": dominant_emotion.capitalize(),
            "confidence": f"{confidence}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def legacy_predict(request: TextRequest):
    return predict_text(request)
