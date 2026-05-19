from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)