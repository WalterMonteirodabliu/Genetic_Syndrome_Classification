import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class FlattenResult:
    X: np.ndarray                 # (n_samples, 320)
    y: np.ndarray                 # (n_samples,)
    meta: pd.DataFrame            # syndrome_id, subject_id, image_id
    syndrome_to_index: Dict[Any, int]
    index_to_syndrome: Dict[int, Any]


def load_pickle(pickle_path: str) -> Dict:
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle não encontrado em: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Estrutura do pickle inesperada: esperado dict no topo.")
    return data


def flatten_hierarchy(data: Dict) -> FlattenResult:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[Dict[str, Any]] = []

    syndrome_ids = sorted(list(data.keys()))
    syndrome_to_index = {sid: i for i, sid in enumerate(syndrome_ids)}
    index_to_syndrome = {i: sid for sid, i in syndrome_to_index.items()}

    skipped = 0

    for syndrome_id, subjects in data.items():
        if not isinstance(subjects, dict):
            continue
        for subject_id, images in subjects.items():
            if not isinstance(images, dict):
                continue
            for image_id, emb in images.items():
                if emb is None:
                    skipped += 1
                    continue

                arr = np.asarray(emb, dtype=np.float32)

                # Integridade: embedding 1D e 320 dims
                if arr.ndim != 1 or arr.shape[0] != 320:
                    skipped += 1
                    continue

                # Integridade: sem NaN/Inf
                if not np.isfinite(arr).all():
                    skipped += 1
                    continue

                X_list.append(arr)
                y_list.append(syndrome_to_index[syndrome_id])
                meta_rows.append(
                    {"syndrome_id": syndrome_id, "subject_id": subject_id, "image_id": image_id}
                )

    if len(X_list) == 0:
        raise ValueError("Nenhuma amostra válida encontrada após validações.")

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int32)
    meta = pd.DataFrame(meta_rows)

    print(f"[flatten] amostras válidas: {len(X_list)} | ignoradas: {skipped}")
    print(f"[flatten] n_síndromes: {len(syndrome_ids)}")
    print(f"[flatten] X shape: {X.shape} | y shape: {y.shape}")

    return FlattenResult(
        X=X,
        y=y,
        meta=meta,
        syndrome_to_index=syndrome_to_index,
        index_to_syndrome=index_to_syndrome,
    )


def compute_eda(result: FlattenResult):
    df = result.meta.copy()
    df["label"] = result.y

    counts = (
        df.groupby("label")
        .size()
        .reset_index(name="n_images")
        .assign(syndrome_id=lambda d: d["label"].map(result.index_to_syndrome))
        .sort_values("n_images", ascending=False)
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        {
            "n_syndromes": [len(result.syndrome_to_index)],
            "n_images_total": [result.X.shape[0]],
            "embedding_dim": [result.X.shape[1]],
            "images_per_syndrome_mean": [counts["n_images"].mean()],
            "images_per_syndrome_median": [counts["n_images"].median()],
            "images_per_syndrome_min": [counts["n_images"].min()],
            "images_per_syndrome_max": [counts["n_images"].max()],
        }
    )

    return summary, counts


def save_eda(summary: pd.DataFrame, counts: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "eda_summary.csv"), index=False)
    counts.to_csv(os.path.join(out_dir, "images_per_syndrome.csv"), index=False)
    print(f"[save] outputs/eda_summary.csv")
    print(f"[save] outputs/images_per_syndrome.csv")