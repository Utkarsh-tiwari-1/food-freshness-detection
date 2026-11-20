# Freshkeep: Food Freshness & Intelligent Shelf-Life Prediction

A small full-stack prototype for visual food-freshness classification and temperature-aware quality-window estimates. The interface supports image upload and mobile camera capture; the API is built with FastAPI; the model is a transfer-learned ResNet18 in PyTorch.

## What is included

- Responsive HTML, CSS, and JavaScript interface for image upload, camera capture, food category, and storage temperature.
- FastAPI endpoints for health, available options, and image prediction.
- ResNet18 training and checkpoint loading for the `fresh`, `aging`, and `spoiled` classes.
- A configurable temperature adjustment and illustrative remaining quality-window estimate.
- Input checks for image type, dimensions, and upload size.

## Start the app

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. The interface loads without a checkpoint, but prediction stays unavailable until a model has been trained and saved. API documentation is at http://127.0.0.1:8000/docs.

## Train a model

Arrange labeled images like this:

```text
data/
  train/
    aging/
    fresh/
    spoiled/
  val/
    aging/
    fresh/
    spoiled/
```

Keep images from the same physical food item in only one split. Then run:

```powershell
python train.py --data data --epochs 8 --batch-size 32
```

Training uses ImageNet-pretrained ResNet18 weights by default, fine-tunes the final residual block and classification head, prints train/validation accuracy each epoch, and saves the best validation checkpoint to `models/freshness_resnet18.pth`. The initial pretrained weights are downloaded by torchvision if they are not cached. Use `--no-pretrained` to train without that download. Restart the API after training.

The checkpoint path and hyperparameters can be changed with `--output`, `--learning-rate`, and `--workers`. Use a separate, representative dataset and inspect per-class behavior before relying on predictions.

## Estimate behavior and limitations

The shelf-life output is an **illustrative quality-window estimate**, not a safe-to-eat determination or a calibrated food-safety prediction. The reference days in `freshness/shelf_life.py` are configurable prototype defaults, not official storage limits. Temperature modifies those defaults with a simplified Q10-style curve; it does not account for elapsed time, packaging, handling, initial age, or the specific variety of food. A photo cannot detect harmful bacteria. Keep refrigerators at 40°F (4°C) or below and follow package dates and authoritative guidance. See the [FDA refrigerator guidance](https://www.fda.gov/food/buy-store-serve-safe-food/refrigerator-thermometers-cold-facts-about-food-safety) and [USDA FSIS cold storage guidance](https://ask.usda.gov/s/article/What-are-suggested-storage-times-for-beef).

No dataset or trained checkpoint is included, so this repository does not claim the 95.63% training / 95.15% validation accuracy from a resume description. Accuracy depends on the dataset and must be measured from a reproducible training run.
