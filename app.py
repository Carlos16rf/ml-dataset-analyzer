from flask import Flask, request, jsonify, render_template
import pandas as pd

from analyze import analyze_dataset, compute_viability_score
import io, base64
import matplotlib.pyplot as plt

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "template"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.json.ensure_ascii = False  # para que salgan tildes bien
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ✅ Ruta HOME: sirve el HTML
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ✅ Ruta health: solo para comprobar que la API vive
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ✅ Ruta analyze: recibe CSV y devuelve JSON
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo con el campo 'file'."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Solo se aceptan archivos .csv"}), 400

    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"No se pudo leer el CSV: {str(e)}"}), 400

    summary, issues, recs = analyze_dataset(df)
    score, grade = compute_viability_score(summary)
    plots = {}

    target = summary.get("target")
    ptype = summary.get("problem_type")

    # -----------------------
    # REGRESIÓN: plots útiles del target
    # -----------------------
    if target and ptype == "regression":
        # 1) Histograma del TARGET
        fig = plt.figure()
        df[target].dropna().plot(kind="hist", bins=50)
        plt.title(f"Distribución del target (regresión): {target}")
        plt.xlabel(target)
        plt.ylabel("frecuencia")
        plots["target_histogram"] = fig_to_base64(fig)

        # 2) Scatter target vs feature más correlacionada
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target in num_cols:
            feature_candidates = [c for c in num_cols if c != target]
            if feature_candidates:
                corr_series = df[feature_candidates + [target]].corr(numeric_only=True)[target].drop(labels=[target])
                best_feat = corr_series.abs().idxmax()
                best_corr = float(corr_series[best_feat])

                plot_df = df[[best_feat, target]].dropna()
                if len(plot_df) > 5000:
                    plot_df = plot_df.sample(5000, random_state=42)

                fig = plt.figure()
                plt.scatter(plot_df[best_feat], plot_df[target], s=8)
                plt.title(f"{target} vs {best_feat} (corr={best_corr:.2f})")
                plt.xlabel(best_feat)
                plt.ylabel(target)
                plots["target_vs_best_feature"] = fig_to_base64(fig)

    # (Opcional) histograma de Amount si existe
    if "Amount" in df.columns:
        fig = plt.figure()
        df["Amount"].dropna().plot(kind="hist", bins=50)
        plt.title("Histograma: Amount")
        plots["amount_histogram"] = fig_to_base64(fig)

    # ✅ Return SIEMPRE dentro de la función
    return jsonify({
        "summary": summary,
        "issues": issues,
        "recommendations": recs,
        "final_score": {
            "viability_score": score,
            "grade": grade
        },
        "plots": plots
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
