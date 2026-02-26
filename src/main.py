import argparse

from .data_processing import load_pickle, flatten_hierarchy, compute_eda, save_eda
from .visualization import run_tsne
from .cross_validation import run_cv_knn, plot_roc_comparison

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pickle_path", type=str, default="data/mini_gm_public_v0.1.p")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()

    data = load_pickle(args.pickle_path)
    result = flatten_hierarchy(data)

    summary, counts = compute_eda(result)

    print("\n[EDA SUMMARY]")
    print(summary.to_string(index=False))

    print("\n[TOP 10 syndromes por #imagens]")
    print(counts.head(10).to_string(index=False))

    # t-SNE plot
    run_tsne(result.X, result.y, args.out_dir)

    save_eda(summary, counts, args.out_dir)


    n_classes = len(result.syndrome_to_index)

    # CV KNN Euclidean
    res_euc, best_k_euc, roc_fpr, roc_tpr_euc = run_cv_knn(
        result.X, result.y, n_classes=n_classes, metric="euclidean", out_dir=args.out_dir
    )

    # CV KNN Cosine
    res_cos, best_k_cos, roc_fpr2, roc_tpr_cos = run_cv_knn(
        result.X, result.y, n_classes=n_classes, metric="cosine", out_dir=args.out_dir
    )

    # ROC plot (best k de cada métrica)
    plot_roc_comparison(
        roc_fpr,
        roc_tpr_euc,
        roc_tpr_cos,
        out_path=f"{args.out_dir}/roc_knn_microavg.png"
    )

    # Tabela final resumida (best ks)
    import pandas as pd

    best_summary = pd.DataFrame([
        {
            "metric": "euclidean",
            "best_k": best_k_euc,
            "auc_macro_ovr_mean": float(res_euc.loc[res_euc["k"] == best_k_euc, "auc_macro_ovr_mean"].iloc[0]),
            "f1_macro_mean": float(res_euc.loc[res_euc["k"] == best_k_euc, "f1_macro_mean"].iloc[0]),
            "top_1_mean": float(res_euc.loc[res_euc["k"] == best_k_euc, "top_1_mean"].iloc[0]),
            "top_3_mean": float(res_euc.loc[res_euc["k"] == best_k_euc, "top_3_mean"].iloc[0]),
            "top_5_mean": float(res_euc.loc[res_euc["k"] == best_k_euc, "top_5_mean"].iloc[0]),
        },
        {
            "metric": "cosine",
            "best_k": best_k_cos,
            "auc_macro_ovr_mean": float(res_cos.loc[res_cos["k"] == best_k_cos, "auc_macro_ovr_mean"].iloc[0]),
            "f1_macro_mean": float(res_cos.loc[res_cos["k"] == best_k_cos, "f1_macro_mean"].iloc[0]),
            "top_1_mean": float(res_cos.loc[res_cos["k"] == best_k_cos, "top_1_mean"].iloc[0]),
            "top_3_mean": float(res_cos.loc[res_cos["k"] == best_k_cos, "top_3_mean"].iloc[0]),
            "top_5_mean": float(res_cos.loc[res_cos["k"] == best_k_cos, "top_5_mean"].iloc[0]),
        }
    ])

    best_summary.to_csv(f"{args.out_dir}/knn_best_summary.csv", index=False)
    print(f"[save] {args.out_dir}/knn_best_summary.csv")

if __name__ == "__main__":
    main()