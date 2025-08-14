# src/predict.py
import re, json, sys, joblib, pandas as pd, numpy as np

ART = joblib.load("models/champion_lr_v1.joblib")
MODEL, G2C = ART["model"], ART["grade2code"]

def _clean(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _frame(rows):
    recs = []
    for r in rows:
        txt = _clean(r.get("text",""))
        words = txt.split()
        charged = str(r.get("charged_to_client","")).upper() in ("YES","Y","TRUE","1")
        grade  = str(r.get("grade","")).title()
        recs.append({
            "text_clean": txt,
            "Worked Time": float(r.get("worked_time", 0.0)),
            "charged_bin": int(charged),
            "grade_enc": int(G2C.get(grade, 0)),
            "low_info": int(len(words) <= 3),
        })
    return pd.DataFrame.from_records(recs)

def predict(rows, topk=3):
    X = _frame(rows)
    if hasattr(MODEL, "predict_proba"):
        P = MODEL.predict_proba(X)
        C = MODEL.classes_
        idx = np.argsort(P, axis=1)[:, -topk:]
        top = [{C[j]: float(P[i, j]) for j in idx[i]} for i in range(len(X))]
        return MODEL.predict(X).tolist(), top
    # fallback to margins
    M = np.atleast_2d(MODEL.decision_function(X))
    C = MODEL.classes_
    idx = np.argsort(M, axis=1)[:, -topk:]
    top = [{C[j]: float(M[i, j]) for j in idx[i]} for i in range(len(X))]
    return MODEL.predict(X).tolist(), top

if __name__ == "__main__":
    rows = json.loads(sys.stdin.read())  # read a JSON list from stdin
    preds, top = predict(rows, topk=3)
    print(json.dumps({"preds": preds, "top": top}, indent=2))
