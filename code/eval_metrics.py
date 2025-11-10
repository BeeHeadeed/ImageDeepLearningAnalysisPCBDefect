import os
import json
import shutil


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

def sample_prediction(OUTPUTS_DIR, SAMPLE_PRED_DIR):
    pred_sample = os.path.join(OUTPUTS_DIR, "baseline", "val_batch0_pred.jpg")
    dst_sample = os.path.join(SAMPLE_PRED_DIR, "pred_1.jpg")
    if os.path.exists(pred_sample):
        shutil.copy(pred_sample, dst_sample)

def save_metrics_to_json(results, METRICS_DIR,model_name):
    try:
        metrics_data = {
            "precision": results.metrics.get("metrics/precision(B)"),
            "F1 Score": results.metrics.get("metrics/F1(B)"),
            "recall": results.metrics.get("metrics/recall(B)"),
            "map50": results.metrics.get("metrics/mAP50(B)"),

        }
        metrics_path = os.path.join(METRICS_DIR, "results_baseline.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
    except Exception as e:
        print(" Could not extract metrics automatically:", e)

    print("Training complete.")
    print(f"Metrics saved to: {os.path.join(METRICS_DIR, f'{model_name}.json')}")

def export_results(OUTPUTS_DIR, SAMPLE_PRED_DIR, METRICS_DIR,results,model_name):
    os.makedirs(os.path.join(OUTPUTS_DIR,model_name), exist_ok=True)
    OUTPUTS_DIR = os.path.join(OUTPUTS_DIR,model_name)
    training_curves(OUTPUTS_DIR,model_name)
    confusion_matrix(OUTPUTS_DIR,model_name)
    sample_prediction(OUTPUTS_DIR, SAMPLE_PRED_DIR)
    save_metrics_to_json(results, METRICS_DIR,model_name)