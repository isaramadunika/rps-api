from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence percentage")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
