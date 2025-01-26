import numpy as np
import scipy.stats as ss
import scipy.spatial.distance as spd

def compute_distance(x, y, metric='gini', nu=2, p=2):
    """Calcul de distance selon la métrique choisie."""
    if metric == 'gini':
        rank_x = ss.rankdata(x, method="average")
        decum_rank_x = len(x) - rank_x + 1
        rank_y = ss.rankdata(y, method="average")
        decum_rank_y = len(y) - rank_y + 1
        return -np.sum((x - y) * (decum_rank_x ** (nu - 1) - decum_rank_y ** (nu - 1)))
    elif metric == 'minkowski':
        return np.sum(np.abs(x - y) ** p) ** (1 / p)
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return spd.cosine(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def init_bounds_dense_python(X, centers, center_half_distances, labels, upper_bounds, lower_bounds, metric='gini', nu=2, p=2):
    """Initialisation des bornes pour chaque échantillon (dense)."""
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    for i in range(n_samples):
        best_cluster = 0
        min_dist = compute_distance(X[i], centers[0], metric=metric, nu=nu, p=p)
        lower_bounds[i, 0] = min_dist
        for j in range(1, n_clusters):
            if min_dist > center_half_distances[best_cluster, j]:
                dist = compute_distance(X[i], centers[j], metric=metric, nu=nu, p=p)
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        labels[i] = best_cluster
        upper_bounds[i] = min_dist

def compute_distance_sparse(x_data, x_indices, center, metric='gini', nu=2, p=2):
    """Calcul de la distance pour des données creuses (sparse) selon la métrique."""
    x_dense = np.zeros(len(center))
    x_dense[x_indices] = x_data
    return compute_distance(x_dense, center, metric=metric, nu=nu, p=p)

def init_bounds_sparse_python(X, centers, center_half_distances, labels, upper_bounds, lower_bounds, metric='gini', nu=2, p=2):
    """Initialisation des bornes pour chaque échantillon (sparse)."""
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    for i in range(n_samples):
        best_cluster = 0
        min_dist = compute_distance_sparse(X.data[X.indptr[i]:X.indptr[i+1]],
                                           X.indices[X.indptr[i]:X.indptr[i+1]],
                                           centers[0], metric=metric, nu=nu, p=p)
        lower_bounds[i, 0] = min_dist
        for j in range(1, n_clusters):
            if min_dist > center_half_distances[best_cluster, j]:
                dist = compute_distance_sparse(X.data[X.indptr[i]:X.indptr[i+1]],
                                               X.indices[X.indptr[i]:X.indptr[i+1]],
                                               centers[j], metric=metric, nu=nu, p=p)
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        labels[i] = best_cluster
        upper_bounds[i] = min_dist

def elkan_iter_chunked_dense_python(X, sample_weight, centers_old, centers_new, weight_in_clusters,
                             center_half_distances, distance_next_center, upper_bounds,
                             lower_bounds, labels, center_shift, metric='gini', nu=2, p=2, update_centers=True):
    """Itération Elkan K-means pour les données denses avec diverses distances."""
    n_samples = X.shape[0]
    n_clusters = centers_new.shape[0]
    n_features = X.shape[1]

    if n_samples == 0:
        return

    for i in range(n_samples):
        upper_bound = upper_bounds[i]
        bounds_tight = False
        label = labels[i]

        if not distance_next_center[label] >= upper_bound:
            for j in range(n_clusters):
                if (j != label and upper_bound > lower_bounds[i, j]
                        and upper_bound > center_half_distances[label, j]):
                    if not bounds_tight:
                        upper_bound = compute_distance(X[i], centers_old[label], metric=metric, nu=nu, p=p)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = True
                    if upper_bound > lower_bounds[i, j] or upper_bound > center_half_distances[label, j]:
                        distance = compute_distance(X[i], centers_old[j], metric=metric, nu=nu, p=p)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance
            labels[i] = label
            upper_bounds[i] = upper_bound

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label, k] += X[i, k] * sample_weight[i]


def elkan_iter_chunked_sparse_python(X, sample_weight, centers_old, centers_new, weight_in_clusters,
                              center_half_distances, distance_next_center, upper_bounds,
                              lower_bounds, labels, center_shift, metric='gini', nu=2, p=2, update_centers=True):
    """Itération Elkan K-means pour les données creuses avec diverses distances."""
    n_samples = X.shape[0]
    n_clusters = centers_new.shape[0]

    if n_samples == 0:
        return

    for i in range(n_samples):
        upper_bound = upper_bounds[i]
        bounds_tight = False
        label = labels[i]

        if not distance_next_center[label] >= upper_bound:
            for j in range(n_clusters):
                if (j != label and upper_bound > lower_bounds[i, j]
                        and upper_bound > center_half_distances[label, j]):
                    if not bounds_tight:
                        upper_bound = compute_distance_sparse(X.data[X.indptr[i]:X.indptr[i+1]],
                                                              X.indices[X.indptr[i]:X.indptr[i+1]],
                                                              centers_old[label], metric=metric, nu=nu, p=p)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = True
                    if upper_bound > lower_bounds[i, j] or upper_bound > center_half_distances[label, j]:
                        distance = compute_distance_sparse(X.data[X.indptr[i]:X.indptr[i+1]],
                                                           X.indices[X.indptr[i]:X.indptr[i+1]],
                                                           centers_old[j], metric=metric, nu=nu, p=p)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance
            labels[i] = label
            upper_bounds[i] = upper_bound

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(X.indptr[i], X.indptr[i+1]):
                centers_new[label, X.indices[k]] += X.data[k] * sample_weight[i]