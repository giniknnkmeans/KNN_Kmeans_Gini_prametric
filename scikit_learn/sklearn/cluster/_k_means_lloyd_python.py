import numpy as np
import scipy.stats as ss
import scipy.spatial.distance as spd
from ._k_means_common_python import _relocate_empty_clusters_dense
from ._k_means_common_python import _relocate_empty_clusters_sparse
from ._k_means_common_python import _average_centers, _center_shift
from sklearn.utils.extmath import row_norms
import scipy.stats as ss

def _gini_distance_centroid(x, Y, nu, rank_centroid, ranks_x):
    """Calcule la distance de Gini entre un point x et des centroïdes Y en utilisant les rangs moyens pour le centroïde."""
    x = np.asarray(x).reshape(1, -1)  
    Y = np.asarray(Y)   
    decum_rank_x = ranks_x ** (nu - 1)  
    decum_rank_centroid = rank_centroid ** (nu - 1)  
    distance = -np.sum((x - Y) * (decum_rank_x - decum_rank_centroid), axis=1)
    return distance


def HasD(x, y):
    total = 0
    for xi, yi in zip(x, y):
        min_value = min(xi, yi)
        max_value = max(xi, yi)
        total += 1  # we sum the 1 in both cases
        if min_value >= 0:
            total -= (1 + min_value) / (1 + max_value)
        else:
            total -= 1 / (1 + max_value + abs(min_value))
    return total

def hellinger_distance(p, q):
    p = np.abs(p)
    q = np.abs(q)
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

# Distance Pearson Chi-Squared
def pearson_chi2(x, y):
    return np.sum((x - y) ** 2 / (x + y))

# Distance Vicis Symmetric 1
def vicis_symmetric_1(x, y):
    total = 0
    for xi, yi in zip(x, y):
        if min(xi, yi) != 0:
            total += (xi - yi) ** 2 / (min(xi, yi) ** 2)
        else:
            total += (xi - yi) ** 2  # Handle the case when min(xi, yi) == 0
    return total


def compute_distance(x, y, metric='gini', nu=2, p=2, rank_centroid=None, ranks_x=None):
    if metric == 'gini':
        return _gini_distance_centroid(x, y, nu=nu, rank_centroid=rank_centroid, ranks_x=ranks_x)
    elif metric == 'minkowski':
        return np.sum(np.abs(x - y) ** p) ** (1 / p)
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return spd.cosine(x, y)
    elif metric == 'canberra':
        return spd.canberra(x, y)
    elif metric == 'hellinger':
        return hellinger_distance(x,y)
    elif metric == 'jensen_shannon':
        return spd.jensenshannon(x, y)
    elif metric == 'hassanat':
        return HasD(x,y)
    elif metric=="pearson_chi2":
        return pearson_chi2(x,y)
    elif metric=="vicis_symmetric_1":
        return vicis_symmetric_1(x,y)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _update_chunk_dense_python(X, sample_weight, centers_old, centers_squared_norms, labels, centers_new, weight_in_clusters, update_centers, metric="gini", nu=2, p=3):
    """K-means combined EM step for one dense data chunk."""
    n_samples, n_features = X.shape
    n_clusters = centers_old.shape[0]

    if metric == "gini":
        X_cat = np.concatenate((X, centers_old), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_ = X_cat.shape[0] - ranks + 1
        ranks = (ranks_ / X_cat.shape[0] * X.shape[0])
        decum_ranks_x = ranks[:X.shape[0]]
        decum_ranks_centroid = ranks[X.shape[0]:]

    pairwise_distances = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            if metric == "gini":
                pairwise_distances[i, j] = compute_distance(X[i], centers_old[j], metric=metric, nu=nu, p=p, rank_centroid=decum_ranks_centroid[j], ranks_x=decum_ranks_x[i])
            else:
                pairwise_distances[i, j] = compute_distance(X[i], centers_old[j], metric=metric, nu=nu, p=p)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i, 0]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i, j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            centers_new[label] += X[i] * sample_weight[i]


def lloyd_iter_chunked_dense_python(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels, center_shift, update_centers=True, metric="gini", nu=2, p=3):
    """Single iteration of K-means lloyd algorithm with dense input."""
    CHUNK_SIZE = 256
    n_samples, n_features = X.shape
    n_clusters = centers_old.shape[0]

    if n_samples == 0:
        return

    centers_squared_norms = row_norms(centers_old, squared=True)

    n_samples_chunk = min(CHUNK_SIZE, n_samples)
    n_chunks = n_samples // n_samples_chunk
    n_samples_rem = n_samples % n_samples_chunk

    if update_centers:
        centers_new.fill(0)
        weight_in_clusters.fill(0)

    for chunk_idx in range(n_chunks + (n_samples_rem > 0)):
        start = chunk_idx * n_samples_chunk
        end = start + (n_samples_rem if chunk_idx == n_chunks else n_samples_chunk)

        _update_chunk_dense_python(
            X[start:end],
            sample_weight[start:end],
            centers_old,
            centers_squared_norms,
            labels[start:end],
            centers_new,
            weight_in_clusters,
            update_centers,
            metric=metric,
            p=p,
            nu=nu
        )

    if update_centers:
        _relocate_empty_clusters_dense(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels)
        _average_centers(centers_new, weight_in_clusters)
        _center_shift(centers_old, centers_new, center_shift, metric=metric, p=p, nu=nu)


def lloyd_iter_chunked_sparse_python(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels, center_shift, update_centers=True, metric="gini", p=3, nu=2):
    """Single iteration of K-means lloyd algorithm with sparse input."""
    CHUNK_SIZE=256
    n_samples = X.shape[0]
    n_clusters = centers_old.shape[0]

    if n_samples == 0:
        return

    n_samples_chunk = min(CHUNK_SIZE, n_samples)
    n_chunks = n_samples // n_samples_chunk
    n_samples_rem = n_samples % n_samples_chunk

    centers_squared_norms = row_norms(centers_old, squared=True)

    if update_centers:
        centers_new.fill(0)
        weight_in_clusters.fill(0)

    for chunk_idx in range(n_chunks + (n_samples_rem > 0)):
        start = chunk_idx * n_samples_chunk
        end = start + (n_samples_rem if chunk_idx == n_chunks else n_samples_chunk)

        # Utilisation de _gini_distance_centroid avec rangs moyens des centroïdes
        _update_chunk_sparse_python(
            X.data[X.indptr[start]: X.indptr[end]],
            X.indices[X.indptr[start]: X.indptr[end]],
            X.indptr[start:end+1],
            sample_weight[start:end],
            centers_old,
            centers_squared_norms,
            labels[start:end],
            centers_new,
            weight_in_clusters,
            update_centers,
            metric=metric,
            nu=nu,
            p=p
        )

    if update_centers:
        _relocate_empty_clusters_sparse(X.data, X.indices, X.indptr, sample_weight, centers_old, centers_new, weight_in_clusters, labels, metric=metric, nu=nu, p=p)
        _average_centers(centers_new, weight_in_clusters, metric=metric)
        _center_shift(centers_old, centers_new, center_shift, metric=metric,nu=nu, p=p)

def _update_chunk_sparse_python(X_data, X_indices, X_indptr, sample_weight, centers_old, centers_squared_norms, labels, centers_new, weight_in_clusters, update_centers, metric="gini", nu=2, p=3):
    """K-means combined EM step for one sparse data chunk."""
    n_samples = len(labels)
    n_clusters = centers_old.shape[0]
    max_floating = np.finfo(float).max

    if metric == "gini":
        X_cat = np.concatenate((X_data, centers_old), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_ = X_cat.shape[0] - ranks + 1
        ranks = (ranks_ / X_cat.shape[0] * X_data.shape[0])
        decum_ranks_x = ranks[:X_data.shape[0]]
        decum_ranks_centroid = ranks[X_data.shape[0]:]

    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            if metric == 'gini':
                sq_dist = _gini_distance_centroid(X_data[X_indptr[i]:X_indptr[i + 1]], centers_old[j], nu=nu, rank_centroid=decum_ranks_centroid[j], ranks_x=decum_ranks_x[i])
            else:
                sq_dist = compute_distance(X_data[X_indptr[i]:X_indptr[i + 1]], centers_old[j], metric=metric, nu=nu, p=p)

            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X_indptr[i], X_indptr[i + 1]):
                centers_new[label, X_indices[k]] += X_data[k] * sample_weight[i]
