"""Engineering data analysis: load forecasting and equipment monitoring.

Run: python run_cases.py
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.exists() else ImageFont.load_default(size=size)


def chart(path, title, series, ylabel, note):
    width, height = 1400, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 30), title, font=font(30), fill="#162D4A")
    left, top, right, bottom = 110, 130, 1330, 610
    values = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    lo, hi = float(values.min()), float(values.max())
    margin = max((hi - lo) * .08, 1)
    lo, hi = lo - margin, hi + margin
    for k in range(6):
        y = bottom - k * (bottom - top) / 5
        v = lo + k * (hi - lo) / 5
        draw.line((left, y, right, y), fill="#E4EAF0", width=2)
        draw.text((15, y - 12), f"{v:.0f}", font=font(20), fill="#546578")
    draw.line((left, top, left, bottom, right, bottom), fill="#546578", width=2)
    for index, (name, vals, color) in enumerate(series):
        points = [(left + i * (right-left) / max(len(vals)-1, 1), bottom - (float(v)-lo)*(bottom-top)/(hi-lo)) for i,v in enumerate(vals)]
        draw.line(points, fill=color, width=3)
        lx = 110 + index * 380
        draw.line((lx, 92, lx+42, 92), fill=color, width=5)
        draw.text((lx+54, 76), name, font=font(22), fill="#344B61")
    draw.text((left, bottom+15), "时间顺序 / Time order", font=font(22), fill="#546578")
    draw.text((70, height-80), ylabel, font=font(21), fill="#546578")
    draw.text((70, height-45), note, font=font(20), fill="#546578")
    img.save(path)


def error_metrics(actual, pred):
    delta = np.asarray(pred)-np.asarray(actual)
    return {
        "MAE_kW": float(np.mean(np.abs(delta))),
        "RMSE_kW": float(np.sqrt(np.mean(delta**2))),
        "MAPE_percent": float(np.mean(np.abs(delta)/np.maximum(np.abs(actual),1e-8))*100),
    }


def load_forecast():
    rng = np.random.default_rng(20261008)
    n = 180 * 24
    dt = pd.date_range("2026-01-01", periods=n, freq="h")
    i = np.arange(n)
    hour = dt.hour.to_numpy()
    weekend = (dt.dayofweek.to_numpy() >= 5).astype(float)
    temp = 22 + 7*np.sin(2*np.pi*i/(24*180)-.8) + 3*np.sin(2*np.pi*(hour-8)/24) + rng.normal(0,1.6,n)
    cooling = np.maximum(temp-24,0)
    heating = np.maximum(18-temp,0)
    load = 1000 + 150*np.sin(2*np.pi*(hour-7)/24) + 70*np.cos(4*np.pi*hour/24) - 150*weekend + 27*cooling + 15*heating + rng.normal(0,38,n)
    raw = pd.DataFrame({"timestamp":dt,"temperature_C":temp,"load_kW":load,"weekend":weekend})
    raw.to_csv(DATA/"load_synthetic.csv", index=False)
    df = raw.copy()
    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)
    df["hour_sin_2"] = np.sin(4*np.pi*hour/24)
    df["hour_cos_2"] = np.cos(4*np.pi*hour/24)
    df["cooling"] = cooling
    df["heating"] = heating
    df["lag_24"] = df.load_kW.shift(24)
    df["lag_168"] = df.load_kW.shift(168)
    df = df.dropna().reset_index(drop=True)
    features = ["hour_sin","hour_cos","hour_sin_2","hour_cos_2","weekend","cooling","heating","lag_24","lag_168"]
    train, test = df.iloc[:-720], df.iloc[-720:]
    mean = train[features].mean().to_numpy()
    scale = train[features].std(ddof=0).to_numpy()
    scale = np.maximum(scale, 1e-12)
    X_train = np.column_stack([np.ones(len(train)),(train[features].to_numpy()-mean)/scale])
    X_test = np.column_stack([np.ones(len(test)),(test[features].to_numpy()-mean)/scale])
    penalty = np.eye(X_train.shape[1])
    penalty[0,0] = 0
    alpha = 1.0
    coef = np.linalg.solve(X_train.T@X_train+alpha*penalty,X_train.T@train.load_kW.to_numpy())
    pred = X_test@coef
    baseline = test.lag_24.to_numpy()
    actual = test.load_kW.to_numpy()
    metrics = {
        "all_records":n,"valid_feature_records":len(df),"train_records":len(train),"test_records":len(test),
        "split":"chronological: final 30 days as test", "seed":20261008,"ridge_alpha":alpha,
        "protocol":"rolling one-hour-ahead; previous observed loads and a supplied simulated weather covariate",
        "model":error_metrics(actual,pred),"lag24_baseline":error_metrics(actual,baseline),
        "features":features,"train_start":str(train.timestamp.iloc[0]),"train_end":str(train.timestamp.iloc[-1]),
        "test_start":str(test.timestamp.iloc[0]),"test_end":str(test.timestamp.iloc[-1]),
    }
    metrics["MAE_improvement_percent"] = 100*(1-metrics["model"]["MAE_kW"]/metrics["lag24_baseline"]["MAE_kW"])
    pd.DataFrame({"timestamp":test.timestamp,"actual_kW":actual,"ridge_prediction_kW":pred,"lag24_baseline_kW":baseline}).to_csv(RESULTS/"load_predictions.csv", index=False)
    np.savez(RESULTS/"load_model.npz",mean=mean,scale=scale,coef=coef,features=np.array(features),alpha=alpha)
    chart(RESULTS/"load_forecast.png","小时负荷预测：评估时段末72小时",[("实际负荷",actual[-72:],"#1C3553"),("岭回归预测",pred[-72:],"#D97706"),("前一天同小时基线",baseline[-72:],"#159B97")],"单位：kW","单小时滚动预测 | 日周期、天气响应与历史负荷特征 | 评估数据为仿真样本")
    return metrics


def equipment_anomaly():
    rng = np.random.default_rng(20261009)
    n = 3000
    dt = pd.date_range("2026-08-01",periods=n,freq="10min")
    idx = np.arange(n)
    speed = 1480+45*np.sin(2*np.pi*idx/144)+rng.normal(0,12,n)
    load_ratio = np.clip(.6+.15*np.sin(2*np.pi*idx/288)+rng.normal(0,.045,n),.2,.95)
    shared = rng.normal(0,1,n)
    vibration = 1.2+.0008*(speed-1480)+.6*load_ratio+.14*shared+rng.normal(0,.16,n)
    temperature = 35+18*load_ratio+.008*(speed-1480)+.4*shared+rng.normal(0,1.9,n)
    current = 6+10*load_ratio+.3*shared+rng.normal(0,.7,n)
    labels = np.zeros(n,dtype=int)
    selected = rng.choice(np.arange(2000,n),size=120,replace=False)
    labels[selected] = 1
    types = np.full(n,"normal",dtype=object)
    for k,t in enumerate(selected):
        if k%3 == 0:
            vibration[t] += rng.uniform(.7,1.7)
            types[t] = "vibration"
        elif k%3 == 1:
            temperature[t] += rng.uniform(5.5,12)
            types[t] = "thermal"
        else:
            vibration[t] += rng.uniform(.4,.9)
            temperature[t] += rng.uniform(3,7)
            current[t] += rng.uniform(1.8,3.6)
            types[t] = "compound"
    df = pd.DataFrame({"timestamp":dt,"speed_rpm":speed,"load_ratio":load_ratio,"vibration_mm_s":vibration,"temperature_C":temperature,"current_A":current,"fault_label":labels,"fault_type":types})
    df.to_csv(DATA/"equipment_synthetic.csv",index=False)
    design = np.column_stack([np.ones(n),speed-1480,load_ratio])
    sensors = df[["vibration_mm_s","temperature_C","current_A"]].to_numpy()
    # The initial 2000 simulated rows are deliberately generated as normal calibration data.
    coef = np.linalg.lstsq(design[:2000],sensors[:2000],rcond=None)[0]
    residual = sensors-design@coef
    mean = residual[:2000].mean(axis=0)
    covariance = np.cov(residual[:2000],rowvar=False)+np.eye(3)*1e-6
    inverse = np.linalg.inv(covariance)
    centered = residual-mean
    scores = np.einsum("ij,jk,ik->i",centered,inverse,centered)
    threshold = float(np.quantile(scores[:2000],.995))
    truth = labels[2000:]
    prediction = (scores[2000:]>threshold).astype(int)
    tp = int(np.sum((prediction==1)&(truth==1)))
    fp = int(np.sum((prediction==1)&(truth==0)))
    fn = int(np.sum((prediction==0)&(truth==1)))
    tn = int(np.sum((prediction==0)&(truth==0)))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    metrics = {"all_records":n,"train_records":2000,"test_records":1000,"test_anomalies":int(truth.sum()),"seed":20261009,"threshold_quantile":.995,"threshold":threshold,"precision":precision,"recall":recall,"F1":2*precision*recall/(precision+recall),"false_positive_rate":fp/(fp+tn),"TP":tp,"FP":fp,"FN":fn,"TN":tn,"protocol":"normal-only initial calibration, subsequent held-out 1000 records; labels only used for evaluation"}
    scored = df.iloc[2000:].copy()
    scored["score"] = scores[2000:]
    scored["threshold"] = threshold
    scored["predicted_anomaly"] = prediction
    scored.to_csv(RESULTS/"equipment_scores.csv",index=False)
    np.savez(RESULTS/"equipment_model.npz",regression_coef=coef,residual_mean=mean,inverse_covariance=inverse,threshold=threshold)
    chart(RESULTS/"equipment_anomaly.png","设备状态监测：独立评估集1000条",[("异常得分",scores[2000:],"#D97706"),("训练集确定的阈值",np.full(1000,threshold),"#159B97")],"得分：残差马氏距离的平方","模型与阈值仅使用前2000条正常数据；标签仅用于测试集评估。")
    return metrics


def main():
    results = {"notice":"Engineering simulation: deterministic synthetic data; metrics describe held-out evaluation only.","runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},"case_1":load_forecast(),"case_2":equipment_anomaly()}
    (RESULTS/"metrics.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
