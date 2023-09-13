from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
from donut_module.donut import initialize_processor, load_model, run_prediction
import pdb
from PIL import Image
from torchvision import transforms as T

app = FastAPI()

processor = initialize_processor()
model = load_model()

@app.post("/donut_prediction/")
async def upload_image(file: UploadFile):

    image = Image.open(file.file).convert('RGB')

    # Rescale image
    transform = T.Resize((525,350)) #1.5 ratio
    image = transform(image)

    prediction = run_prediction(image, model, processor)

    return {'prediction': prediction}
