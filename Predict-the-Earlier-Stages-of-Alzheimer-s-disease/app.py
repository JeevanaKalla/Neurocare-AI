from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- COGNITIVE SCREENING PAGE ----------------
@app.route("/cognitive")
def cognitive():
    return render_template("cognitive.html")


# ---------------- BIOMARKER PAGE ----------------
@app.route("/biomarker")
def biomarker():
    return render_template("biomarker.html")


# ---------------- FULL TEST PAGE ----------------
@app.route("/full")
def full():
    return render_template("full.html")


# =========================================================
#                  PROCESSING ROUTES
# =========================================================


# ---------------- COGNITIVE RESULT ----------------
@app.route("/predict_cognitive", methods=["POST"])
def predict_cognitive():

    q1 = int(request.form["q1"])
    q2 = int(request.form["q2"])
    q3 = int(request.form["q3"])
    q4 = int(request.form["q4"])
    q5 = int(request.form["q5"])

    score = q1 + q2 + q3 + q4 + q5
    risk = (score / 10) * 100

    if risk < 30:
        label = "🟢 Low Cognitive Risk"
        advice = "Normal memory function. Maintain healthy lifestyle."
    elif risk < 60:
        label = "🟡 Mild Cognitive Risk (MCI)"
        advice = "Monitor memory changes and consider medical consultation."
    else:
        label = "🔴 High Cognitive Risk"
        advice = "Strong signs of cognitive impairment. Clinical evaluation recommended."

    return render_template(
        "result.html",
        label=label,
        advice=advice,
        biomarker_score="N/A",
        questionnaire_score=round(risk, 2),
        final_score=round(risk, 2),
        mode="Cognitive Screening"
    )


# ---------------- BIOMARKER RESULT ----------------
@app.route("/predict_biomarker", methods=["POST"])
def predict_biomarker():

    features = np.array([[
        float(request.form["Cystatin_C"]),
        float(request.form["MMP10"]),
        float(request.form["tau"])
    ]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[0]
        risk = float(np.max(prob) * 100)
    else:
        risk = 50 if prediction == 1 else 20

    if risk < 30:
        label = "🟢 Low Biomarker Risk"
        advice = "Biomarkers within normal range."
    elif risk < 60:
        label = "🟡 Mild Biomarker Abnormality"
        advice = "Possible early biological changes detected."
    else:
        label = "🔴 High Biomarker Risk"
        advice = "Strong Alzheimer’s biomarker signal detected."

    return render_template(
        "result.html",
        label=label,
        advice=advice,
        biomarker_score=round(risk, 2),
        questionnaire_score="N/A",
        final_score=round(risk, 2),
        mode="Biomarker Analysis"
    )


# ---------------- FULL HYBRID RESULT ----------------
@app.route("/predict_full", methods=["POST"])
def predict_full():

    # biomarker input
    features = np.array([[
        float(request.form["Cystatin_C"]),
        float(request.form["MMP10"]),
        float(request.form["tau"])
    ]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[0]
        biomarker_risk = float(np.max(prob) * 100)
    else:
        biomarker_risk = 50

    # questionnaire input
    q1 = int(request.form["q1"])
    q2 = int(request.form["q2"])
    q3 = int(request.form["q3"])
    q4 = int(request.form["q4"])
    q5 = int(request.form["q5"])

    questionnaire_score = q1 + q2 + q3 + q4 + q5
    questionnaire_risk = (questionnaire_score / 10) * 100

    # fusion logic
    final_score = (0.7 * biomarker_risk) + (0.3 * questionnaire_risk)

    if final_score < 30:
        label = "🟢 Low Alzheimer’s Risk"
        advice = "No major signs detected. Maintain brain health."
    elif final_score < 60:
        label = "🟡 Mild Cognitive Impairment (MCI)"
        advice = "Early warning signs detected. Monitor regularly."
    else:
        label = "🔴 High Alzheimer’s Risk"
        advice = "Strong risk indicators. Clinical evaluation advised."

    return render_template(
        "result.html",
        label=label,
        advice=advice,
        biomarker_score=round(biomarker_risk, 2),
        questionnaire_score=round(questionnaire_risk, 2),
        final_score=round(final_score, 2),
        mode="Full AI Assessment"
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)