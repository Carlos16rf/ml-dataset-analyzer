from flask import Flask, request, jsonify, render_template
import pandas as pd

from analyze import analyze_dataset, compute_viability_score
import io, base64
import matplotlib.pyplot as plt

app = Flask(__name__)
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

    # Plot 1: Missing values (top 10)
    null_pct = df.isna().mean().sort_values(ascending=False)
    top_null = null_pct[null_pct > 0].head(10)
    if len(top_null) > 0:
        fig = plt.figure()
        (top_null * 100).plot(kind="bar")
        plt.ylabel("% nulos")
        plt.title("Top columnas con valores nulos")
        plots["missing_values_top10"] = fig_to_base64(fig)

    # Plot 2: Target distribution (si clasificación)
    target = summary.get("target")
    ptype = summary.get("problem_type")
    if target and ptype == "classification":
        vc = df[target].value_counts(dropna=False).head(10)
        fig = plt.figure()
        vc.plot(kind="bar")
        plt.ylabel("conteo")
        plt.title(f"Distribución de target: {target}")
        plots["target_distribution"] = fig_to_base64(fig)

    # Plot 3: Histograma de variable numérica
    num_cols = df.select_dtypes(include="number").columns.tolist()
    hist_col = None

    if "Amount" in df.columns:
        hist_col = "Amount"
    elif len(num_cols) > 0:
        hist_col = num_cols[0]

    if hist_col:
        fig = plt.figure()
        df[hist_col].dropna().plot(kind="hist", bins=50)
        plt.title(f"Histograma: {hist_col}")
        plots["histogram"] = fig_to_base64(fig)

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
