from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import os
import joblib

# ---------- Matplotlib (headless) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Plotly ----------
import plotly.express as px

app = Flask(__name__)

# ---------- CONFIG ----------
COLOR_COLS = {"u_g", "g_r", "r_i", "i_z"}
MAG_COLS = {"u", "g", "r", "i", "z"}

MAX_FILE_SIZE_MB = 100
MAX_PLOT_POINTS = 10_000

AMBIGUITY_THRESHOLD = 0.10
OOD_THRESHOLD = 0.45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_astro_data.csv")

SCATTER_PATH = os.path.join(STATIC_DIR, "ug_gr_scatter.png")

# ---------- LOAD ML ARTIFACTS ----------
model = joblib.load(os.path.join(BASE_DIR, "rf_star_classifier.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
FEATURES = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# ---------- HELPERS ----------
def normalize_columns(cols):
    out = []
    for c in cols:
        c = c.replace("\ufeff", "").strip().lower()
        c = re.sub(r"[^\w]", "", c)
        out.append(c)
    return out

# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def home():
    validation_message = None
    validation_status = None
    detected_columns = None
    total_objects = None

    plot_ready = False
    plotly_html = None
    plotly_3d_html = None

    mean_confidence = None
    ambiguous_percent = None
    ood_percent = None
    class_distribution = None

    if request.method == "POST":
        action = request.form.get("action")
        df = None

        # ---------- LOAD DATA ----------
        if action == "upload":
            file = request.files.get("csv_file")
            if file and file.filename.endswith(".csv"):
                df = pd.read_csv(file, encoding="utf-8-sig")

        elif action == "sample":
            df = pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8-sig")

        if df is not None:
            df.columns = normalize_columns(df.columns)
            total_objects = len(df)

            # ---------- FEATURE CHECK ----------
            if not COLOR_COLS.issubset(df.columns):
                if MAG_COLS.issubset(df.columns):
                    df["u_g"] = df["u"] - df["g"]
                    df["g_r"] = df["g"] - df["r"]
                    df["r_i"] = df["r"] - df["i"]
                    df["i_z"] = df["i"] - df["z"]
                else:
                    return render_template(
                        "index.html",
                        validation_status="error",
                        validation_message="Dataset missing required photometric columns."
                    )

            validation_status = "ok"
            validation_message = "Dataset valid for astro analysis ✅"
            detected_columns = df.columns.tolist()

            # ---------- ML INFERENCE ----------
            X = df[FEATURES]
            X_scaled = scaler.transform(X)

            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)

            confidence = np.max(y_proba, axis=1)
            top2 = np.sort(y_proba, axis=1)[:, -2:]
            is_ambiguous = (top2[:, 1] - top2[:, 0]) < AMBIGUITY_THRESHOLD

            labels = label_encoder.inverse_transform(y_pred)

            final_labels = []
            for i, lbl in enumerate(labels):
                if is_ambiguous[i]:
                    idx = np.argsort(y_proba[i])[-2:]
                    names = label_encoder.inverse_transform(idx)
                    final_labels.append(f"{names[1]} / {names[0]} (Ambiguous)")
                else:
                    final_labels.append(lbl)

            df["predicted_class"] = final_labels
            df["confidence"] = confidence

            # ---------- STEP 3B: VISUAL STATUS ----------
            df["visual_status"] = "Confident"
            df.loc[confidence < OOD_THRESHOLD, "visual_status"] = "Out-of-Distribution"
            df.loc[
                (df["visual_status"] == "Confident") &
                (df["predicted_class"].str.contains("Ambiguous")),
                "visual_status"
            ] = "Ambiguous"

            # ---------- STEP 3C: METRICS ----------
            mean_confidence = round(float(np.mean(confidence)), 3)
            ambiguous_percent = round((df["visual_status"] == "Ambiguous").mean() * 100, 2)
            ood_percent = round((df["visual_status"] == "Out-of-Distribution").mean() * 100, 2)
            class_distribution = df["predicted_class"].value_counts().to_dict()

            # ---------- SAMPLING ----------
            df_plot = df.sample(MAX_PLOT_POINTS, random_state=42) if len(df) > MAX_PLOT_POINTS else df

            # ---------- STATIC SCATTER ----------
            plt.figure(figsize=(6, 5))
            plt.scatter(df_plot["u_g"], df_plot["g_r"], s=10, alpha=0.6)
            plt.xlabel("u − g")
            plt.ylabel("g − r")
            plt.title("u−g vs g−r")
            plt.tight_layout()
            plt.savefig(SCATTER_PATH, dpi=150)
            plt.close()

            # ---------- PLOTLY 2D ----------
            fig2d = px.scatter(
                df_plot,
                x="u_g",
                y="g_r",
                color="visual_status",
                title="u−g vs g−r (ML Confidence Map)",
                color_discrete_map={
                    "Confident": "#00cc96",
                    "Ambiguous": "#EF553B",
                    "Out-of-Distribution": "#636EFA"
                }
            )
            fig2d.update_layout(template="plotly_dark")
            plotly_html = fig2d.to_html(full_html=False)

            # ---------- PLOTLY 3D ----------
            fig3d = px.scatter_3d(
                df_plot,
                x="u_g",
                y="g_r",
                z="r_i",
                color="visual_status",
                title="3D Color Space (Uncertainty Aware)",
                color_discrete_map={
                    "Confident": "#00cc96",
                    "Ambiguous": "#EF553B",
                    "Out-of-Distribution": "#636EFA"
                }
            )
            fig3d.update_layout(template="plotly_dark")
            plotly_3d_html = fig3d.to_html(full_html=False)

            plot_ready = True

    return render_template(
        "index.html",
        validation_message=validation_message,
        validation_status=validation_status,
        detected_columns=detected_columns,
        total_objects=total_objects,
        mean_confidence=mean_confidence,
        ambiguous_percent=ambiguous_percent,
        ood_percent=ood_percent,
        class_distribution=class_distribution,
        plot_ready=plot_ready,
        plotly_html=plotly_html,
        plotly_3d_html=plotly_3d_html
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
