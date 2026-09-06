"""
app.py
Nutrifit food detection + nutrition lookup (JSON test version).

Detects all food items in a photo using the trained YOLO model, then looks
up nutrition info (calories, protein, carbs, fat) for each detected item
from a local JSON file, and prints a per-item + total summary.

This is a JSON-based stand-in for the MongoDB lookup that will replace
`load_nutrition_db()` / `lookup_nutrition()` later - the rest of the
pipeline (detection, aggregation, display) stays the same either way.

Usage:
    python app.py                       -> opens a file picker dialog
    python app.py path/to/image.jpg      -> uses the given image directly
"""

import sys
import os
import json
from ultralytics import YOLO
MODEL_PATH = r"C:\Users\Andrew\NutriFitAiModel\model\yolov8n_nutrifit_167\weights\best.pt"
NUTRITION_JSON_PATH = "nutrition_data.json"
CONF_THRESHOLD = 0.40 


def pick_image_with_dialog():
    """Open a native file picker dialog and return the selected image path (or None)."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select a food photo to analyze",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path or None


def load_nutrition_db(path):
    """Load the nutrition lookup table from JSON (a list of food records) and
    index it by name for fast lookup.

    Expected record format:
        {"id": int, "name": str, "serving": str,
         "calories_kcal": number, "protein_g": number,
         "carbs_g": number, "fats_g": number}

    TODO (MongoDB swap-in): replace this with a MongoDB query, e.g.:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client["nutrifit"]
        collection = db["foods"]
        # then in lookup_nutrition(), do: collection.find_one({"name": class_name})
    """
    if not os.path.exists(path):
        print(f"⚠️ Nutrition data file not found at: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {record["name"]: record for record in records}


def lookup_nutrition(class_name, nutrition_db):
    """Look up nutrition info for a detected class name.
    Returns None if not found in the database.
    """
    return nutrition_db.get(class_name)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        sys.exit(1)

    nutrition_db = load_nutrition_db(NUTRITION_JSON_PATH)
    print(f"📖 Loaded nutrition data for {len(nutrition_db)} food items")

    if len(sys.argv) >= 2:
        source = sys.argv[1]
    else:
        print("📂 Opening file picker...")
        source = pick_image_with_dialog()
        if not source:
            print("❌ No file selected. Exiting.")
            sys.exit(0)

    if not os.path.exists(source):
        print(f"❌ Path not found: {source}")
        sys.exit(1)

    print(f"📦 Loading model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    print(f"🔍 Analyzing: {source}\n")
    results = model.predict(
        source=source,
        conf=CONF_THRESHOLD,
        save=True,
        project="runs_nutrifit",
        name="app_predictions",
        exist_ok=True,
    )

    for r in results:
        img_name = os.path.basename(r.path)
        print(f"{'='*70}")
        print(f"📷 {img_name}")
        print(f"{'='*70}\n")

        if len(r.boxes) == 0:
            print("No food items detected. Try a clearer photo or lower CONF_THRESHOLD.\n")
            continue

        detected_items = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detected_items.append((cls_name, conf))

        totals = {"calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}
        missing_nutrition = []

        print(f"{'Food':<25} {'Confidence':>10}  {'Serving':<18} {'Cal':>6} {'Protein':>8} {'Carbs':>7} {'Fat':>6}")
        print("-" * 90)

        for cls_name, conf in detected_items:
            info = lookup_nutrition(cls_name, nutrition_db)
            if info is None:
                print(f"{cls_name:<25} {conf:>9.1%}  (no nutrition data yet)")
                missing_nutrition.append(cls_name)
                continue

            serving = info.get("serving", "")
            print(f"{cls_name:<25} {conf:>9.1%}  {serving:<18} {info['calories_kcal']:>6} {info['protein_g']:>7}g {info['carbs_g']:>6}g {info['fats_g']:>5}g")
            totals["calories_kcal"] += info["calories_kcal"]
            totals["protein_g"] += info["protein_g"]
            totals["carbs_g"] += info["carbs_g"]
            totals["fats_g"] += info["fats_g"]

        print("-" * 90)
        print(f"{'TOTAL':<25} {'':>10}  {'':<18} {totals['calories_kcal']:>6} {totals['protein_g']:>7.1f}g {totals['carbs_g']:>6.1f}g {totals['fats_g']:>5.1f}g\n")

        if missing_nutrition:
            print(f"⚠️ No nutrition data yet for: {', '.join(set(missing_nutrition))}")
            print("   Add these to nutrition_data.json to include them in totals.\n")

    print(f"🖼️ Annotated image saved to: runs_nutrifit/app_predictions/")


if __name__ == "__main__":
    main()