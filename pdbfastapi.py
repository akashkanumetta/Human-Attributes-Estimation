from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

model = load_model(r"C:\Users\msdak\Desktop\vscode\height_weight_gender_age_model.h5", custom_objects={"mse": mse})  

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224)) 
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)[0]

    height = predictions[0] 
    weight = predictions[1] 
    gender = "Male" if round(predictions[2]) == 1 else "Female"

    return {
        "height": round(float(height), 2),
        "weight": round(float(weight), 2),
        "gender": gender
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)