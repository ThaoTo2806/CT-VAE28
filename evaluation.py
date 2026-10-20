import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def evaluation(y_true, y_pred):
    """Evaluate clustering with Hungarian matching, including missing clusters."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.metrics import adjusted_rand_score

    y_true = np.asarray(y_true).reshape(-1).copy()
    y_pred = np.asarray(y_pred).reshape(-1).copy()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("Empty labels")

    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)

    nmi = normalized_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    )
    ari = adjusted_rand_score(y_true, y_pred)

    # Contingency matrix: rows = true classes, columns = predicted clusters
    contingency = np.zeros(
        (len(true_classes), len(pred_classes)), dtype=np.int64
    )
    for i, true_label in enumerate(true_classes):
        for j, pred_label in enumerate(pred_classes):
            contingency[i, j] = np.sum(
                (y_true == true_label) & (y_pred == pred_label)
            )

    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Map matched clusters to true labels; unmatched clusters remain distinct.
    mapped = np.full(y_pred.shape, fill_value=None, dtype=object)
    mapping = {
        pred_classes[col]: true_classes[row]
        for row, col in zip(row_ind, col_ind)
    }
    for pred_label in pred_classes:
        mapped[y_pred == pred_label] = mapping.get(pred_label, "__unmatched__")

    acc = accuracy_score(y_true.astype(str), mapped.astype(str))
    f1 = f1_score(
        y_true.astype(str),
        mapped.astype(str),
        labels=true_classes.astype(str),
        average="macro",
        zero_division=0,
    )
    return acc, nmi, ari, f1

def eva(k, labels, emb):
    embeddings = emb.cpu().data.numpy()
    model = KMeans(n_clusters=k, n_init=20)
    cluster_id = model.fit_predict(embeddings)
    acc, nmi, ari, f1 = evaluation(labels, cluster_id)
    return acc, nmi, ari, f1

def end2end_eva(labels, cluster_id):
    acc, nmi, ari, f1 = evaluation(labels, cluster_id)
    return acc, nmi, ari, f1