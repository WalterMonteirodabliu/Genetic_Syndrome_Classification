import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from .metrics import (
    confusion_matrix_np,
    f1_macro_from_cm,
    topk_accuracy,
    auc_macro_ovr,
    roc_micro_average,
)


def make_knn(k: int, metric: str) -> KNeighborsClassifier:
    # metric: "euclidean" ou "cosine"
    if metric == "cosine":
        return KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",   # recomendado p/ cosine
            weights="distance"   # geralmente ajuda
        )
    return KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="distance"
    )


def evaluate_fold(y_true, y_pred, proba, n_classes, topk_list=(1, 3, 5)):
    cm = confusion_matrix_np(y_true, y_pred, n_classes)
    f1 = f1_macro_from_cm(cm)
    auc = auc_macro_ovr(y_true, proba, n_classes)
    topk = {f"top_{k}": topk_accuracy(y_true, proba, k) for k in topk_list}
    return {"f1_macro": f1, "auc_macro_ovr": auc, **topk}


def run_cv_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    metric: str,
    k_values=range(1, 16),
    n_splits: int = 10,
    random_state: int = 42,
    out_dir: str = "outputs",
):
    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results_rows = []
    roc_fpr_grid = np.linspace(0, 1, 200)
    roc_tprs_by_k = {}  # k -> list[tpr_interp_per_fold]

    for k in k_values:
        fold_metrics = []
        fold_tprs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = make_knn(k, metric)
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)  # (n_test, n_classes)
            y_pred = np.argmax(proba, axis=1)

            m = evaluate_fold(y_test, y_pred, proba, n_classes)
            fold_metrics.append(m)

            # ROC micro-average da fold
            fpr, tpr = roc_micro_average(y_test, proba, n_classes)
            tpr_interp = np.interp(roc_fpr_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            fold_tprs.append(tpr_interp)

        # agrega folds
        df_f = pd.DataFrame(fold_metrics)
        row = {"metric": metric, "k": k}
        for col in df_f.columns:
            row[f"{col}_mean"] = float(df_f[col].mean())
            row[f"{col}_std"] = float(df_f[col].std(ddof=1))
        results_rows.append(row)

        roc_tprs_by_k[k] = fold_tprs

    results = pd.DataFrame(results_rows).sort_values(["metric", "k"]).reset_index(drop=True)

    # escolhe melhor k por AUC
    best_idx = results["auc_macro_ovr_mean"].idxmax()
    best_k = int(results.loc[best_idx, "k"])

    # ROC médio do best_k
    mean_tpr = np.mean(roc_tprs_by_k[best_k], axis=0)
    mean_tpr[-1] = 1.0

    # salva tabela
    table_path = os.path.join(out_dir, f"knn_cv_results_{metric}.csv")
    results.to_csv(table_path, index=False)
    print(f"[save] {table_path}")
    print(f"[best] metric={metric} best_k={best_k} auc={results.loc[best_idx, 'auc_macro_ovr_mean']:.4f}")

    return results, best_k, roc_fpr_grid, mean_tpr


def plot_roc_comparison(
    roc_fpr,
    euclid_tpr,
    cosine_tpr,
    out_path: str
):
    plt.figure(figsize=(8, 6))
    plt.plot(roc_fpr, euclid_tpr, label="Euclidean (micro-avg)")
    plt.plot(roc_fpr, cosine_tpr, label="Cosine (micro-avg)")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Micro-Average) - KNN")
    plt.legend()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[save] {out_path}")