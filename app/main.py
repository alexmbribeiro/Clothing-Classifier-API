from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import gradio as gr
from model import load_model, predict


# ----------------------------
# Load model
# ----------------------------
model = load_model()

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="Clothing Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    class_name = predict(model, image)
    return {"class": class_name}

# ----------------------------
# Gradio interface
# ----------------------------
def gradio_predict(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))  # convert numpy → PIL
    return predict(model, img)

iface = gr.Interface(fn=gradio_predict, inputs="image", outputs="text",
                     title="Clothing Classifier",
                     description="Upload an image of clothing to classify it.")

# Launch Gradio in same script
if __name__ == "__main__":
    import uvicorn
    import threading

    # Start FastAPI server in a thread
    def start_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    threading.Thread(target=start_api, daemon=True).start()

    # Start Gradio UI
    iface.launch(server_name="0.0.0.0", server_port=7860)
