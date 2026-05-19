from fastapi import FastAPI

from backend.app.api.routes import router


app = FastAPI(
    title="SmartDefectAI",
    description="Industrial Defect Detection API",
    version="1.0"
)


app.include_router(router)


@app.get("/")
def home():

    return {
        "message": "SmartDefectAI API Running"
    }