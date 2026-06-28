from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.response import PredictionResponse
from app.utils.preprocessing import ImagePreprocessor

router = APIRouter()
preprocessor = ImagePreprocessor()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile | None = File(None)) -> PredictionResponse:
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded")

        image_tensor = preprocessor.preprocess(contents)
        predictor = request.app.state.predictor
        result = predictor.predict(image_tensor)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc
