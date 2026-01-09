from analyze import analyze_dataset,compute_viability_score
import sys 
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze.py <ruta_csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    summary, issues, recs = analyze_dataset(df)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== ISSUES ===")
    for it in issues:
        print("-", it)

    print("\n=== RECOMMENDATIONS ===")
    for r in recs:
        print("-", r)
    score, grade = compute_viability_score(summary)

    print("\n=== FINAL SCORE ===")
    print(f"Viability score: {score}/100  |  Grade: {grade}")
