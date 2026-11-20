from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from freshness.config import ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES, MODEL_PATH, ROOT_DIR
from freshness.model import FreshnessModel
from freshness.shelf_life import BASE_DAYS_AT_4C, estimate_shelf_life


app = FastAPI(
    title="Food Freshness & Shelf-Life API",
    description="Image-based freshness classification with a temperature-aware quality-window estimate.",
    version="1.0.0",
)
predictor = FreshnessModel(MODEL_PATH)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": predictor.ready,
        "model_error": predictor.error,
        "device": str(predictor.device),
    }


@app.get("/api/options")
def options() -> dict[str, object]:
    return {
        "food_types": list(BASE_DAYS_AT_4C),
        "freshness_classes": ["fresh", "aging", "spoiled"],
        "max_image_bytes": MAX_IMAGE_BYTES,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    food_type: str = Form("produce"),
    temperature_c: float = Form(4.0),
) -> dict[str, object]:
    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="The trained ResNet18 model is not available. Follow the README training steps and restart the API.",
        )
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    if not -20 <= temperature_c <= 60:
        raise HTTPException(status_code=422, detail="Temperature must be between -20 and 60 C.")

    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
    try:
        image = Image.open(BytesIO(contents))
        if image.width * image.height > 25_000_000:
            raise HTTPException(status_code=413, detail="Image dimensions are too large to process.")
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from None
    finally:
        await file.close()

    if food_type not in BASE_DAYS_AT_4C:
        food_type = "other"

    result = predictor.predict(image)
    estimate = estimate_shelf_life(result.label, food_type, temperature_c)
    return {
        "freshness": result.label,
        "confidence": round(result.confidence, 4),
        "probabilities": result.probabilities,
        "food_type": food_type,
        "temperature_c": temperature_c,
        "shelf_life_days": {
            "low": estimate.low_days,
            "high": estimate.high_days,
            "unit": "days",
        },
        "estimate_note": estimate.message,
        "safety_note": "An image prediction cannot confirm food safety. Follow date labels, storage history, and official food-safety guidance.",
    }
