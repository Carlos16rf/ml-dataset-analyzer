import pandas as pd

TARGET_HINTS = {"target", "label", "y", "class", "outcome", "output", "response"}
NEGATIVE_HINTS = {"gender", "sex", "city", "country", "state", "zip", "postal", "email", "name"}
ID_HINTS = {"id", "uuid", "code", "number", "index"}

def descubrir(df):
    n_rows = len(df)
    cols = list(df.columns)

    # 1) Por nombre típico (alta confianza)
    lower_map = {c.lower(): c for c in cols}
    for hint in TARGET_HINTS:
        if hint in lower_map:
            col = lower_map[hint]
            return col, f"coincide con hint de target ('{hint}')", "alta"

    # 2) Scoring de candidatos
    best_col = None
    best_score = float("-inf")
    best_reason = ""
    best_conf = "baja"

    for col in cols:
        s = df[col]
        col_lower = str(col).lower()

        nun = s.nunique(dropna=True)
        if nun <= 1:
            continue  # constante -> no puede ser target útil

        # descartar IDs claros
        ratio_unique = nun / n_rows if n_rows > 0 else 0
        if ratio_unique > 0.98 and any(h in col_lower for h in ID_HINTS):
            continue

        score = 0
        reasons = []

        # 2.1) Penalizar nombres típicos de feature (reduce falsos positivos como gender)
        if any(h in col_lower for h in NEGATIVE_HINTS):
            score -= 3
            reasons.append("nombre típico de feature (penaliza)")

        # 2.2) Bonus si está al final (muchos datasets ponen la target al final)
        pos = cols.index(col)
        if pos == len(cols) - 1:
            score += 2
            reasons.append("está en la última columna (+2)")
        elif pos >= len(cols) - 3:
            score += 1
            reasons.append("está cerca del final (+1)")

        # 2.3) Clasificación candidata: pocos únicos
        if nun <= 20:
            score += 3
            reasons.append(f"baja cardinalidad (nunique={nun}) (+3)")
        # 2.4) Regresión candidata: numérica y muchos únicos pero no casi-única
        else:
            if pd.api.types.is_numeric_dtype(s):
                if ratio_unique < 0.98:
                    score += 2
                    reasons.append(f"numérica y alta cardinalidad (nunique={nun}) (+2)")
                else:
                    # casi única pero sin hint de ID: sospechosa
                    score -= 2
                    reasons.append("casi única por fila (posible ID) (-2)")
            else:
                # muchas categorías no numéricas (p.ej. textos)
                score -= 1
                reasons.append("no numérica con alta cardinalidad (-1)")

        # 2.5) Bonus extra si su nombre sugiere resultado (aunque no sea exacto)
        if any(k in col_lower for k in {"result", "score", "price", "amount", "cost", "outcome", "response"}):
            score += 1
            reasons.append("nombre sugiere salida/resultado (+1)")

        # 2.6) Selección del mejor
        if score > best_score:
            best_score = score
            best_col = col
            best_reason = "; ".join(reasons) if reasons else "heurística general"

    # 3) Fallback final si no se encontró candidato
    if best_col is None:
        return cols[-1], "fallback: última columna (sin candidatos válidos)", "baja"

    # 4) Confianza según score
    if best_score >= 4:
        conf = "alta"
    elif best_score >= 2:
        conf = "media"
    else:
        conf = "baja"

    return best_col, f"selección por scoring: {best_reason} | score={best_score}", conf



def problem_type(y):
    nun = y.nunique(dropna=True)

    if nun <= 20:
        return "classification", f"porque la variable objetivo tiene pocos valores distintos (nunique={nun}) y no representa una magnitud continua "

    if not pd.api.types.is_numeric_dtype(y):
        return "classification", "target no numérica"

    return "regression", f"target numérica con muchos valores únicos (nunique={nun})"


def balanceo(y):
    counts = y.value_counts(normalize=True, dropna=True)
    k = len(counts)

    # Si hay 0 o 1 clase, no es un problema "clasificable"
    if k <= 1:
        return "no_clasificable", f"Solo {k} clase en la target."

    min_ratio = counts.min()
    ideal = 1.0 / k
    rel = min_ratio / ideal  # 1.0 = perfectamente balanceado

    # Etiquetas por cercanía al ideal
    if rel >= 0.80:
        sev = "balanceado"
    elif rel >= 0.60:
        sev = "ligero"
    elif rel >= 0.40:
        sev = "claro"
    elif rel >= 0.25:
        sev = "severo"
    else:
        sev = "extremo"

    detail = (
        f"{k} clases | minoritaria {min_ratio*100:.2f}% | ideal {ideal*100:.2f}% | "
        f"relación vs ideal {rel:.2f}"
    )
    return sev, detail


def compute_viability_score(summary):
    """
    Score 0-100 (más alto = mejor). Heurística simple basada en contadores en summary.
    """
    score = 100

    # Nulos
    n_null40 = int(summary.get("n_cols_over_40pct_null", 0))
    n_null20 = int(summary.get("n_cols_over_20pct_null", 0))
    score -= 15 * n_null40
    score -= 5 * max(n_null20 - n_null40, 0)

    # Constantes
    n_const = int(summary.get("n_constant_cols", 0))
    score -= 5 * n_const

    # IDs
    n_id = int(summary.get("n_id_like_cols", 0))
    score -= 7 * n_id

    # Leakage por nombre
    n_leak = int(summary.get("n_leak_name_cols", 0))
    score -= 7 * n_leak

    # Desbalance (si clasificación)
    sev = summary.get("imbalance_severity")
    if sev == "ligero":
        score -= 5
    elif sev == "claro":
        score -= 10
    elif sev == "severo":
        score -= 20
    elif sev == "extremo":
        score -= 30

    # Clamp 0..100
    score = max(0, min(100, score))

    # Grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "E"

    return score, grade


def analyze_dataset(df):
    summary = {}
    issues = []
    recs = []

    # 1) Target
    target, target_reason, target_conf  = descubrir(df)
    summary["target"] = target
    summary["target_reason"] = target_reason
    summary["target_confidence"] = target_conf
    # 2) Tipo de problema
    ptype, ptype_reason = problem_type(df[target])
    summary["problem_type"] = ptype
    summary["problem_reason"] = ptype_reason

    # 3) Info básica
    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])

    # -------------------------
    # CHECK: Nulos por columna
    # -------------------------
    null_pct = df.isna().mean()
    summary["n_cols_over_20pct_null"] = int((null_pct > 0.20).sum())
    summary["n_cols_over_40pct_null"] = int((null_pct > 0.40).sum())

    null_pct_sorted = null_pct.sort_values(ascending=False)

    for col, pct in null_pct_sorted.items():
        if pct >= 0.40:
            issues.append(f"Columna '{col}' tiene {pct*100:.1f}% de valores nulos.")
            recs.append(f"Considera eliminar '{col}' o imputar valores (mediana/modelos).")
        elif pct > 0.20:
            issues.append(f"Columna '{col}' tiene {pct*100:.1f}% de valores nulos (moderado).")
            recs.append(f"Evalúa imputación para '{col}' y comprueba su impacto en el modelo.")

    if summary["n_cols_over_20pct_null"] == 0:
        recs.append("No se detectan columnas con un porcentaje relevante de valores nulos.")

    # -------------------------
    # CHECK: Columnas constantes
    # -------------------------
    constant_cols_count = 0
    for col in df.columns:
        nun = df[col].nunique(dropna=True)
        if nun <= 1:
            constant_cols_count += 1
            issues.append(f"Columna '{col}' es constante (nunique={nun}).")
            recs.append(f"Se recomienda eliminar la columna '{col}' porque no aporta información.")

    summary["n_constant_cols"] = int(constant_cols_count)

    if constant_cols_count == 0:
        recs.append("No se detectan columnas constantes en el dataset.")

    # -------------------------
    # CHECK: Columnas tipo ID (casi únicas + nombre sospechoso)
    # -------------------------
    ID_HINTS = {"id", "uuid", "code", "number", "index"}
    n_rows = df.shape[0]
    id_cols_count = 0

    for col in df.columns:
        if col == target:
            continue

        nun = df[col].nunique(dropna=True)
        ratio_nun = nun / n_rows if n_rows > 0 else 0
        col_lower = col.lower()

        if ratio_nun > 0.95 and any(h in col_lower for h in ID_HINTS):
            id_cols_count += 1
            issues.append(f"Columna '{col}' parece un identificador (valores únicos: {ratio_nun*100:.1f}%).")
            recs.append(f"Se recomienda excluir '{col}' del entrenamiento para evitar leakage.")

    summary["n_id_like_cols"] = int(id_cols_count)

    if id_cols_count == 0:
        recs.append("No se detectan columnas que parezcan identificadores o leakage.")

    # -------------------------
    # CHECK: Leakage por nombre (heurística)
    # -------------------------
    LEAK_HINTS = {
        "label", "target", "outcome", "result", "final", "groundtruth", "truth",
        "confirmed", "fraud", "churn", "approved", "status"
    }

    leak_name_cols_count = 0
    target_lower = target.lower()

    for col in df.columns:
        if col == target:
            continue

        col_lower = col.lower()

        if target_lower in col_lower or any(h in col_lower for h in LEAK_HINTS):
            leak_name_cols_count += 1
            issues.append(f"Columna '{col}' podría introducir leakage por su nombre (revísala).")
            recs.append(
                f"Verifica si '{col}' está disponible antes del momento de la predicción; si no, exclúyela."
            )

    summary["n_leak_name_cols"] = int(leak_name_cols_count)

    if leak_name_cols_count == 0:
        recs.append("No se detectan señales claras de leakage por nombre de columna.")

    # -------------------------
    # CHECK: Desbalance (solo clasificación)
    # -------------------------
    if ptype == "classification":
        sev, msg = balanceo(df[target])
        summary["imbalance_severity"] = sev
        summary["imbalance_detail"] = msg

        if sev != "balanceado":
            issues.append(f"Target desbalanceada ({sev}). {msg}")
            recs.append("Usa class_weight / focal loss y evalúa Recall, F1, PR-AUC.")
            recs.append("Considera SMOTE/undersampling y validación estratificada.")
        else:
            recs.append(
                "La target está razonablemente balanceada. "
                "Puedes centrarte en feature engineering, selección de modelos y validación cruzada."
            )

  
    return summary, issues, recs
