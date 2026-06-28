import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.model_loader import ModelLoader
from app.routers.predict import router as predict_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rps_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model during startup...")
    model_loader = ModelLoader(model_path=settings.model_path, classes=settings.class_names)
    app.state.predictor = model_loader.load_predictor()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="Rock Paper Scissors API",
    version="1.0.0",
    description="Production-ready FastAPI service for Rock-Paper-Scissors image classification.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response status: %s", response.status_code)
    return response


app.include_router(predict_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Rock Paper Scissors API Running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
