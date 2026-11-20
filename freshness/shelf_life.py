"""Transparent, configurable quality-window estimates for the prototype."""

from dataclasses import dataclass


# Illustrative starting points at 4 C, in days. These are not official storage
# limits; real shelf life depends on the exact food, packaging, age, and handling.
BASE_DAYS_AT_4C = {
    "produce": 5.0,
    "dairy": 4.0,
    "meat": 2.0,
    "seafood": 1.5,
    "bakery": 3.0,
    "leftovers": 3.0,
    "other": 3.0,
}

FRESHNESS_MULTIPLIER = {"fresh": 1.0, "aging": 0.35, "spoiled": 0.0}


@dataclass
class ShelfLifeEstimate:
    low_days: float
    high_days: float
    temperature_c: float
    reference_days: float
    message: str


def estimate_shelf_life(label: str, food_type: str, temperature_c: float) -> ShelfLifeEstimate:
    """Estimate a remaining quality window; never interpret this as a safety test."""
    category = food_type if food_type in BASE_DAYS_AT_4C else "other"
    reference = BASE_DAYS_AT_4C[category]
    stage = FRESHNESS_MULTIPLIER.get(label, 0.35)

    # A simplified Q10-style adjustment, capped to avoid extreme forecasts.
    if temperature_c <= 4:
        temperature_factor = min(1.25, 2 ** ((4 - temperature_c) / 20))
    else:
        temperature_factor = 2 ** (-(temperature_c - 4) / 10)

    midpoint = max(0.0, reference * stage * temperature_factor)
    estimate = ShelfLifeEstimate(
        low_days=round(max(0.0, midpoint * 0.75), 1),
        high_days=round(midpoint * 1.25, 1),
        temperature_c=temperature_c,
        reference_days=reference,
        message="",
    )

    if label == "spoiled":
        estimate.low_days = 0.0
        estimate.high_days = 0.0
        estimate.message = "The image model classified this item as spoiled. Do not use this estimate to decide whether food is safe."
    elif temperature_c > 4:
        estimate.message = "Storage above 4 C shortens this illustrative quality window. Check storage time and official food-safety guidance."
    else:
        estimate.message = "Illustrative quality-window estimate based on the image class and reported temperature."
    return estimate
