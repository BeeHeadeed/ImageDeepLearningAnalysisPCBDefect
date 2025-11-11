import os
import json
import shutil
import pandas as pd

def training_curves(OUTPUTS_DIR,model_name):
    src_curves = os.path.join(OUTPUTS_DIR, model_name, "results.png")
    dst_curves = os.path.join(OUTPUTS_DIR, f"{model_name}_training_curves.png")
    if os.path.exists(src_curves):
        shutil.copy(src_curves, dst_curves)

def confusion_matrix(OUTPUTS_DIR,model_name):
    src_confmat = os.path.join(OUTPUTS_DIR,model_name, "confusion_matrix.png")
    dst_confmat = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    if os.path.exists(src_confmat):
        shutil.copy(src_confmat, dst_confmat)


def sample_predictions(OUTPUTS_DIR):
    SAMPLE_PRED_DIR = os.path.join(OUTPUTS_DIR, "sample_predictions")
    os.makedirs(SAMPLE_PRED_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(OUTPUTS_DIR)
    copied = 0
    for file in os.listdir(MODEL_DIR):
        if "batch" in file.lower() and file.lower().endswith((".jpg")):
            src = os.path.join(MODEL_DIR, file)
            dst = os.path.join(SAMPLE_PRED_DIR, f"pred_{file}")
            shutil.move(src, dst)
            copied += 1

    if copied == 0:
        print("No prediction images found in outputs directory.")
    else:
        print(f"Copied {copied} sample prediction(s) to '{SAMPLE_PRED_DIR}'")

def save_metrics_to_json(OUTPUTS_DIR,METRICS_DIR,model_name):
    csv_path = os.path.join(OUTPUTS_DIR,"results.csv")
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    precision = float(last_row.get("metrics/precision(B)", 0))
    recall = float(last_row.get("metrics/recall(B)", 0))
    f1 = 2 * (precision * recall) / (precision + recall )
    map50 = float(last_row.get("metrics/mAP50(B)", 0))
    metrics_data = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50
    }

    json_path = os.path.join(METRICS_DIR, f"results_{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)


def export_results(OUTPUTS_DIR, METRICS_DIR,model_name):
    os.makedirs(os.path.join(OUTPUTS_DIR,model_name), exist_ok=True)
    os.makedirs(os.path.join(METRICS_DIR), exist_ok=True)
    OUTPUTS_DIR = os.path.join(OUTPUTS_DIR,model_name)
    training_curves(OUTPUTS_DIR,model_name)
    confusion_matrix(OUTPUTS_DIR,model_name)
    sample_predictions(OUTPUTS_DIR)
    save_metrics_to_json(OUTPUTS_DIR,METRICS_DIR,model_name)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    METRICS_DIR = os.path.join(BASE_DIR, "metrics")

    export_results(OUTPUTS_DIR, METRICS_DIR,"baseline")

    export_results(OUTPUTS_DIR, METRICS_DIR,"enhanced_exp1")