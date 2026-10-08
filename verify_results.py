from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
m = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
load = pd.read_csv(root / "results" / "load_predictions.csv")
delta = load.ridge_prediction_kW - load.actual_kW
assert len(load) == m["case_1"]["test_records"] == 720
calculated = {
    "MAE_kW": float(np.abs(delta).mean()),
    "RMSE_kW": float(np.sqrt((delta ** 2).mean())),
    "MAPE_percent": float((np.abs(delta) / np.abs(load.actual_kW)).mean() * 100),
}
for name, value in calculated.items():
    assert np.isclose(value, m["case_1"]["model"][name]), name
baseline_error = load.lag24_baseline_kW - load.actual_kW
assert np.isclose(np.abs(baseline_error).mean(), m["case_1"]["lag24_baseline"]["MAE_kW"])
eq = pd.read_csv(root / "results" / "equipment_scores.csv")
assert len(eq) == m["case_2"]["test_records"] == 1000
assert np.array_equal(eq.predicted_anomaly, (eq.score > eq.threshold).astype(int))
for name, pred, label in [("TP", 1, 1), ("FP", 1, 0), ("FN", 0, 1), ("TN", 0, 0)]:
    assert int(((eq.predicted_anomaly == pred) & (eq.fault_label == label)).sum()) == m["case_2"][name], name
tp, fp, fn = m["case_2"]["TP"], m["case_2"]["FP"], m["case_2"]["FN"]
assert np.isclose(tp / (tp + fp), m["case_2"]["precision"])
assert np.isclose(tp / (tp + fn), m["case_2"]["recall"])
print("Verified: 720 load predictions and 1000 equipment scores; metrics match per-record results.")
