import numpy as np


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], n_classes), dtype=np.int32)
    oh[np.arange(y.shape[0]), y] = 1
    return oh


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def f1_macro_from_cm(cm: np.ndarray) -> float:
    # F1 por classe: 2*TP/(2*TP+FP+FN), macro avg
    n_classes = cm.shape[0]
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    # y_true em [0..C-1], proba shape (N, C)
    topk = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    hits = (topk == y_true.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())


def auc_binary_rank(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUC binário via Mann–Whitney U (ranking). Sem sklearn.
    y_true_bin: {0,1}
    """
    y_true_bin = y_true_bin.astype(np.int32)
    pos = y_true_bin == 1
    neg = y_true_bin == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # ranks com handling simples de ties (média dos ranks)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # ajuste de ties: média dos ranks nos grupos com mesmo score
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    # U = sum_ranks_pos - n_pos*(n_pos+1)/2
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u / (n_pos * n_neg)
    return float(auc)


def auc_macro_ovr(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        a = auc_binary_rank(y_bin, proba[:, c])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def roc_curve_binary(y_true_bin: np.ndarray, y_score: np.ndarray):
    """
    Retorna fpr, tpr para ROC binária.
    Implementação manual: thresholds em scores ordenados desc.
    """
    y_true_bin = y_true_bin.astype(np.int32)
    desc = np.argsort(y_score)[::-1]
    y_score = y_score[desc]
    y_true_bin = y_true_bin[desc]

    p = int((y_true_bin == 1).sum())
    n = int((y_true_bin == 0).sum())
    if p == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = 0
    fps = 0
    fpr = [0.0]
    tpr = [0.0]

    # percorre e atualiza quando muda threshold
    prev_score = None
    for yt, sc in zip(y_true_bin, y_score):
        if prev_score is None:
            prev_score = sc
        elif sc != prev_score:
            fpr.append(fps / n)
            tpr.append(tps / p)
            prev_score = sc

        if yt == 1:
            tps += 1
        else:
            fps += 1

    # ponto final
    fpr.append(fps / n)
    tpr.append(tps / p)

    return np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)


def roc_micro_average(y_true: np.ndarray, proba: np.ndarray, n_classes: int):
    """
    ROC micro-average (flatten one-hot).
    """
    y_oh = one_hot(y_true, n_classes).reshape(-1)
    scores = proba.reshape(-1)
    return roc_curve_binary(y_oh, scores)