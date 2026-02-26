import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_tsne(X, y, out_dir="outputs", random_state=42):
    print("[t-SNE] Reduzindo dimensionalidade para 2D...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )

    X_2d = tsne.fit_transform(X)

    print("[t-SNE] Concluído.")

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap="tab10",
        alpha=0.7
    )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Syndrome Label")

    plot_path = os.path.join(out_dir, "tsne_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[save] {plot_path}")

    return X_2d