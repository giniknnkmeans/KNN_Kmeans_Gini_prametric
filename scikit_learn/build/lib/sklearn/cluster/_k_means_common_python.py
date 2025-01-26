import numpy as np
from math import sqrt
import scipy.stats as ss
import scipy.spatial.distance as spd

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

def manhattan_distance(a, b, n_features, squared=False):
    """Manhattan distance between a dense and b dense."""
    return np.sum(np.abs(a - b))

def minkowski_distance(a, b, n_features, p=3, squared=False):
    """Minkowski distance between a dense and b dense."""
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def cosine_distance(a, b, n_features, squared=False):
    """Cosine distance between a dense and b dense."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - np.dot(a, b) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

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

def _distance_dense_dense(a, b, n_features, metric="euclidean", p=3, nu=2, squared=False, rank_centroid=None, ranks_x=None):
    """Generic distance function for dense vectors."""
    if metric == "manhattan":
        return manhattan_distance(a, b, n_features, squared)
    elif metric == "minkowski":
        return minkowski_distance(a, b, n_features, p, squared)
    elif metric == "cosine":
        return cosine_distance(a, b, n_features, squared)
    elif metric == "gini":
        return _gini_distance_centroid(a, b, nu=nu, rank_centroid=rank_centroid, ranks_x=ranks_x)
    elif metric == "canberra":
        return spd.canberra(a, b)
    elif metric == 'hellinger':
        return hellinger_distance(a,b)
    elif metric == 'jensen_shannon':
        return spd.jensenshannon(a,b)
    elif metric == 'hassanat':
        return HasD(a,b)
    elif metric=="pearson_chi2":
        return pearson_chi2(a,b)
    elif metric=="vicis_symmetric_1":
        return vicis_symmetric_1(a,b)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _distance_sparse_dense(a_data, a_indices, b, b_squared_norm, metric="euclidean", p=3, squared=False, rank_centroid=None, ranks_x=None, nu=2):
    """Generic distance function for sparse and dense vectors."""
    if metric == "manhattan":
        result = 0.0
        for i in range(a_indices.shape[0]):
            result += abs(a_data[i] - b[a_indices[i]])
        return result
    elif metric == "minkowski":
        result = 0.0
        for i in range(a_indices.shape[0]):
            result += abs(a_data[i] - b[a_indices[i]]) ** p
        return result ** (1 / p)
    elif metric == "cosine":
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            return 0
        result = np.dot(a_data, b[a_indices]) / norm_b
        return 1 - result
    elif metric == "gini":
        return _gini_distance_centroid(a_data, b, rank_centroid=rank_centroid, ranks_x=ranks_x, nu=nu, squared=squared)
    elif metric == "canberra":
        return spd.canberra(a_data, b)
    elif metric == "hellinger":
        return np.sqrt(np.sum((np.sqrt(a_data) - np.sqrt(b)) ** 2)) / np.sqrt(2)
    elif metric == "jensen_shannon":
        return spd.jensenshannon(a_data, b)
    elif metric == "hassanat":
        return np.mean(np.abs(a_data - b) / (a_data + b ))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _inertia_dense_python(X, sample_weight, centers, labels, n_threads, single_label=-1, metric="gini", nu=2, p=3):
    """Compute inertia for dense input data."""
    inertia = 0.0
    n_clusters = centers.shape[0]
    n_features = X.shape[1]
    if metric=="gini":
        """
        ranks_x = np.apply_along_axis(ss.rankdata, 0, X, method="ordinal")
        decum_ranks_x = X.shape[0] - ranks_x + 1

        ranks_centroid = np.apply_along_axis(ss.rankdata, 0, centers, method="ordinal")
        decum_ranks_centroid = centers.shape[0] - ranks_centroid + 1"""
        X_cat = np.concatenate((X, centers), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_= X_cat.shape[0] - ranks + 1
        ranks = (ranks_ / X_cat.shape[0] * X.shape[0])
        decum_ranks_x = ranks[:X.shape[0]]
        decum_ranks_centroid = ranks[X.shape[0]:]
    """
    if metric == "gini":
        # Calcul des rangs pour chaque point
        ranks_x = np.apply_along_axis(ss.rankdata, 0, X, method="ordinal")
        decum_ranks_x = X.shape[0] - ranks_x + 1

        # Calcul des rangs moyens pour chaque centroïde
        rank_centroids = []
        for j in range(n_clusters):
            cluster_points_indices = np.where(labels == j)[0]
            if len(cluster_points_indices) > 0:
                cluster_ranks = ranks_x[cluster_points_indices, :]
                rank_centroids.append(np.mean(cluster_ranks, axis=0))
            else:
                rank_centroids.append(np.zeros(n_features))
        rank_centroids = np.array(rank_centroids)
"""
    # Calcul de l'inertie
    for i in range(X.shape[0]):
        j = labels[i]
        if single_label < 0 or single_label == j:
            if metric == "gini":
                sq_dist = _distance_dense_dense(
                    X[i], centers[j], X.shape[1], metric=metric, p=p, nu=nu,
                    squared=True, ranks_x=decum_ranks_x[i], rank_centroid=decum_ranks_centroid[j])
                #sq_dist = _distance_dense_dense(X[i], centers[j], X.shape[1], metric=metric, p=p, nu=nu, squared=True, rank_centroid=rank_centroids[j], ranks_x=decum_ranks_x[i])
            else:
                sq_dist = _distance_dense_dense(X[i], centers[j], X.shape[1], metric=metric, p=p, squared=True)
            inertia += sq_dist * sample_weight[i]
    
    return inertia


def _inertia_sparse_python(X, sample_weight, centers, labels, n_threads, single_label=-1, metric="gini", nu=2, p=3):
    """Compute inertia for sparse input data."""
    X_data, X_indices, X_indptr = X.data, X.indices, X.indptr
    centers_squared_norms = row_norms(centers, squared=True)
    inertia = 0.0
    n_clusters = centers.shape[0]
    n_features = centers.shape[1]
    if metric == "gini":
        """
        for i in range(X.shape[0]):
            sparse_row = X_data[X_indptr[i]:X_indptr[i + 1]]
            rank_x_data = ss.rankdata(sparse_row, method="ordinal")
            ranks_x.append(X_data[X_indptr[i]:X_indptr[i + 1]].shape[0] - rank_x_data + 1)
            decum_ranks_x = np.array(ranks_x)
        rank_x_centroid = ss.rankdata(centers, method="ordinal")
        decum_ranks_centroid = centers.shape[0] - ranks_centroid + 1"""
        X_cat = np.concatenate((X, centers), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_= X_cat.shape[0] - ranks + 1
        ranks = (ranks_ / X_cat.shape[0] * X.shape[0])
        decum_ranks_x = ranks[:X.shape[0]]
        decum_ranks_centroid = ranks[X.shape[0]:]
    """
    if metric == "gini":
        # Calcul des rangs moyens pour chaque point
        ranks_x = []
        for i in range(X.shape[0]):
            sparse_row = X_data[X_indptr[i]:X_indptr[i + 1]]
            rank_x_data = ss.rankdata(sparse_row, method="ordinal")
            ranks_x.append(X_data[X_indptr[i]:X_indptr[i + 1]].shape[0] - rank_x_data + 1)
        decum_ranks_x = np.array(ranks_x)

        # Calcul des rangs moyens pour chaque centroïde
        rank_centroids = []
        for j in range(n_clusters):
            cluster_indices = np.where(labels == j)[0]
            if len(cluster_indices) > 0:
                cluster_ranks = decum_ranks_x[cluster_indices, :]
                rank_centroids.append(np.mean(cluster_ranks, axis=0))
            else:
                rank_centroids.append(np.zeros(n_features))

        rank_centroids = np.array(rank_centroids)
    """
    for i in range(X.shape[0]):
        j = labels[i]
        if single_label < 0 or single_label == j:
            if metric == "gini":
                sq_dist = _distance_sparse_dense(
                    X_data[X_indptr[i]: X_indptr[i + 1]],
                    X_indices[X_indptr[i]: X_indptr[i + 1]],
                    centers[j],
                    centers_squared_norms[j],
                    metric=metric,
                    p=p,
                    nu=nu,
                    squared=True,
                    rank_centroid=decum_ranks_centroid[j],
                    ranks_x=decum_ranks_x[i]
                )
            else:
                sq_dist = _distance_sparse_dense(
                    X_data[X_indptr[i]: X_indptr[i + 1]],
                    X_indices[X_indptr[i]: X_indptr[i + 1]],
                    centers[j],
                    centers_squared_norms[j],
                    metric=metric,
                    p=p,
                    squared=True
                )
            inertia += sq_dist * sample_weight[i]

    return inertia


def _relocate_empty_clusters_dense(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels, metric="gini", p=3, nu=2):
    """Relocate centers which have no sample assigned to them."""
    empty_clusters = np.where(weight_in_clusters == 0)[0]
    if len(empty_clusters) == 0:
        return
    n_features = X.shape[1]

    if metric == "gini":
        """
        ranks_x = np.apply_along_axis(ss.rankdata, 0, X, method="ordinal")
        decum_ranks_x = X.shape[0] - ranks_x + 1
        ranks_centroid = np.apply_along_axis(ss.rankdata, 0, centers_old, method="ordinal")
        decum_ranks_centroid = centers_old.shape[0] - ranks_centroid + 1"""
        X_cat = np.concatenate((X, centers_old), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_= X_cat.shape[0] - ranks + 1
        ranks = (ranks_ / X_cat.shape[0] * X.shape[0])
        decum_ranks_x = ranks[:X.shape[0]]
        decum_ranks_centroid = ranks[X.shape[0]:]

    """
    if metric == "gini":
        ranks_x = np.apply_along_axis(ss.rankdata, 0, X, method="ordinal")
        decum_ranks_x = X.shape[0] - ranks_x + 1

        rank_centroids = []
        for j in range(centers_old.shape[0]):
            cluster_points_indices = np.where(labels == j)[0]
            if len(cluster_points_indices) > 0:
                cluster_ranks = decum_ranks_x[cluster_points_indices, :]
                rank_centroids.append(np.mean(cluster_ranks, axis=0))
            else:
                rank_centroids.append(np.zeros(n_features)) 
        rank_centroids = np.array(rank_centroids)"""

    
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        j = labels[i]
        if metric == "gini":
            distances[i] = _gini_distance_centroid(X[i], centers_old[j], nu=nu, rank_centroid=decum_ranks_centroid[j], ranks_x= decum_ranks_x[i])
        else:
            distances[i] = _distance_dense_dense(X[i], centers_old[j], X.shape[1], metric=metric, p=p, nu=nu)
        
    far_from_centers = np.argpartition(distances, -len(empty_clusters))[-len(empty_clusters):]

    for idx, new_cluster_id in enumerate(empty_clusters):
        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]
        old_cluster_id = labels[far_idx]

        centers_new[old_cluster_id] -= X[far_idx] * weight
        centers_new[new_cluster_id] = X[far_idx] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight


def _relocate_empty_clusters_sparse(X_data, X_indices, X_indptr, sample_weight, centers_old, centers_new, weight_in_clusters, labels, metric="gini", p=3, nu=2):
    """Relocate centers which have no sample assigned to them."""
    empty_clusters = np.where(weight_in_clusters == 0)[0]
    if len(empty_clusters) == 0:
        return

    n_samples = len(X_indptr) - 1
    centers_squared_norms = row_norms(centers_old, squared=True)

    if metric == "gini":
        # Calcul des rangs pour chaque point
        X_cat = np.concatenate((X_data, centers_old), axis=0)
        ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
        ranks_= X_cat.shape[0] - ranks + 1
        ranks = (ranks_/ X_cat.shape[0] * X_data.shape[0])
        decum_ranks_x = ranks[:X_data.shape[0]]
        decum_ranks_centroid = ranks[X_data.shape[0]:]
        #ranks_x = []
        '''
        for i in range(n_samples):
            sparse_row = X_data[X_indptr[i]:X_indptr[i + 1]]
            rank_x_data = ss.rankdata(sparse_row, method="ordinal")
            ranks_x.append(X_data[X_indptr[i]:X_indptr[i + 1]].shape[0] - rank_x_data + 1)
        decum_ranks_x = np.array(ranks_x)
        ranks_centroid = np.apply_along_axis(ss.rankdata, 0, centers_old, method="ordinal")
        decum_ranks_centroid = centers_old.shape[0] - ranks_centroid + 1'''
        

    """
    if metric == "gini":
        # Calcul des rangs pour chaque point
        ranks_x = []
        for i in range(n_samples):
            sparse_row = X_data[X_indptr[i]:X_indptr[i + 1]]
            rank_x_data = ss.rankdata(sparse_row, method="ordinal")
            ranks_x.append(X_data[X_indptr[i]:X_indptr[i + 1]].shape[0] - rank_x_data + 1)
        decum_ranks_x = np.array(ranks_x)

        # Calcul des rangs moyens des points de chaque cluster
        rank_centroids = []
        for j in range(centers_old.shape[0]):
            cluster_points_indices = np.where(labels == j)[0]
            if len(cluster_points_indices) > 0:
                cluster_ranks = decum_ranks_x[cluster_points_indices, :]
                rank_centroids.append(np.mean(cluster_ranks, axis=0))
            else:
                rank_centroids.append(np.zeros(centers_old.shape[1]))
        rank_centroids = np.array(rank_centroids)
"""
    # Calcul des distances
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        j = labels[i]
        if metric=="gini":
            distances[i] = _distance_sparse_dense(
                    X_data[X_indptr[i]: X_indptr[i + 1]],
                    X_indices[X_indptr[i]: X_indptr[i + 1]],
                    centers_old[j],
                    centers_squared_norms[j],
                    metric=metric,
                    p=p,
                    nu=nu,
                    squared=True,
                    rank_centroid=decum_ranks_centroid[j],
                    ranks_x=decum_ranks_x[i]
                )
        else:
            distances[i] = _distance_sparse_dense(
                X_data[X_indptr[i]:X_indptr[i + 1]],
                X_indices[X_indptr[i]:X_indptr[i + 1]],
                centers_old[j],
                centers_squared_norms[j],
                metric=metric,
                p=p,
                nu=nu
            )


    # Sélection des points les plus éloignés pour relocaliser les clusters vides
    far_from_centers = np.argpartition(distances, -len(empty_clusters))[-len(empty_clusters):]

    for idx, new_cluster_id in enumerate(empty_clusters):
        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]
        old_cluster_id = labels[far_idx]

        for k in range(X_indptr[far_idx], X_indptr[far_idx + 1]):
            centers_new[old_cluster_id, X_indices[k]] -= X_data[k] * weight
            centers_new[new_cluster_id, X_indices[k]] = X_data[k] * weight

        # Mise à jour des centroïdes en fonction des rangs moyens pour Gini
        if metric == "gini":
            centers_new[old_cluster_id] -= rank_centroids[old_cluster_id] * weight
            centers_new[new_cluster_id] = rank_centroids[new_cluster_id] * weight

        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight



def _center_shift(centers_old, centers_new, center_shift, metric="gini", p=3, nu=2):
    """Compute shift between old and new centers."""
    # Calcul des rangs pour les nouveaux centres
    X_cat = np.concatenate((centers_new, centers_old), axis=0)
    ranks = np.apply_along_axis(ss.rankdata, 0, X_cat)
    ranks_= X_cat.shape[0] - ranks + 1
    ranks = (ranks_ / X_cat.shape[0] * centers_new.shape[0])
    decum_ranks_x = ranks[:centers_new.shape[0]]
    decum_ranks_centroid = ranks[centers_new.shape[0]:]
    """
    ranks_x = np.apply_along_axis(ss.rankdata, 0, centers_new, method="ordinal")
    decum_ranks_x = centers_new.shape[0] - ranks_x + 1

    # Calcul des rangs pour les anciens centres
    ranks_centroid = np.apply_along_axis(ss.rankdata, 0, centers_old, method="ordinal")
    decum_ranks_centroid = centers_old.shape[0] - ranks_centroid + 1"""
    
    # Calcul du déplacement des centres
    for j in range(centers_old.shape[0]):
        if metric == "gini":
            # Utilisation de la distance de Gini avec les rangs des centres
            center_shift[j] = _distance_dense_dense(
                centers_new[j], 
                centers_old[j], 
                centers_old.shape[1], 
                metric="gini", 
                p=p, 
                nu=nu, 
                squared=False,
                rank_centroid=decum_ranks_centroid[j],  # Rangs de l'ancien centroïde
                ranks_x=decum_ranks_x[j]  # Rangs du nouveau centroïde
            )
        else:
            # Autres métriques (manhattan, euclidienne, etc.)
            center_shift[j] = _distance_dense_dense(
                centers_new[j], 
                centers_old[j], 
                centers_old.shape[1], 
                metric=metric, 
                p=p, 
                nu=nu, 
                squared=False
            )



def _is_same_clustering_python(labels1, labels2, n_clusters):
    """Check if two arrays of labels are the same up to a permutation of the labels."""
    mapping = np.full(n_clusters, -1, dtype=np.int32)

    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True

def _average_centers(centers, weight_in_clusters):
    """Average new centers wrt weights."""
    for j in range(centers.shape[0]):
        if weight_in_clusters[j] > 0:
            centers[j] /= weight_in_clusters[j]
