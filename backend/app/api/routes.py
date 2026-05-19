import os

from fastapi import APIRouter, UploadFile, File
from backend.app.services.model_service import predict


router = APIRouter()


UPLOAD_DIR = "backend/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/predict")
async def predict_defect(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:

        f.write(await file.read())

    result = predict(file_path)

    return result