# Dataset layout

Place labeled images into this structure before training. Use the same food items and camera conditions across the classes, and split images into train and validation sets before training to avoid leakage.

```text
data/
  train/
    fresh/
    aging/
    spoiled/
  val/
    fresh/
    aging/
    spoiled/
```

The classes describe visible freshness stages, not food safety. Never label images from the same physical item in both splits.
