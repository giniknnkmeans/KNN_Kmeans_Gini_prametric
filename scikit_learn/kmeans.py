from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, fetch_openml
import scipy.stats as ss
import warnings
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score

np.random.seed(42)
def align_clusters(y_true, y_pred):
    """
    Aligns predicted clusters to ground-truth labels using the Hungarian algorithm.
    - Inputs:
        y_true: True labels for the data.
        y_pred: Predicted cluster labels.
    - Returns:
        Aligned cluster labels where predicted clusters are matched to ground-truth labels.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    cluster_to_label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    aligned_labels = [cluster_to_label_mapping[cluster] for cluster in y_pred]
    return aligned_labels

def test(X, y, dataset_name, k, initial_centroids):
    """
    Performs clustering evaluation on a dataset using KMeans with Gini-based distance and Silhouette criteria for the research of nu parameter.
    - Inputs:
        X: Feature matrix of the dataset.
        y: Ground-truth labels.
        dataset_name: Name of the dataset.
        k: Number of clusters to use in KMeans.
    - Returns:
        DataFrame with precision and recall results for various Gini parameters (nu).
    """
    #5fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    nu_values = np.arange(1.1, 5.1, 0.1)
    best_results = {
        'Gini_generalized': {'Precision': 0, 'Recall': 0, 'Nu': 0},
        'Gini_generalized_silhouette': {'Precision': 0, 'Recall': 0, 'Nu': 0},
        'Gini': {'Precision': 0, 'Recall': 0},
        'Euclidean': {'Precision': 0, 'Recall': 0},
        'Manhattan': {'Precision': 0, 'Recall': 0},
        'Minkowski': {'Precision': 0, 'Recall': 0, 'p': 3},
        'Cosine': {'Precision': 0, 'Recall': 0},
        'Canberra': {'Precision': 0, 'Recall': 0},
        'Hellinger': {'Precision': 0, 'Recall': 0},
        'Jensen_Shannon': {'Precision': 0, 'Recall': 0},
        'Pearson_Chi2': {'Precision': 0, 'Recall': 0},
        'Vicis_Symmetric_1': {'Precision': 0, 'Recall': 0},
        'Hassanat': {'Precision': 0, 'Recall': 0},
    }

    precision_results = {distance: [] for distance in best_results.keys()}
    recall_results = {distance: [] for distance in best_results.keys()}
    steps_results = {distance: [] for distance in best_results.keys()}
    best_nu_values = []
    best_nu_per_fold = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if not np.issubdtype(y_train.dtype, np.number):
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

        # Search the best nu parameter that maximized the Silhouette criteria
        best_precision = 0
        best_nu = 2
        best_nu_silhouette = None
        best_silhouette_score = -1

        for nu in nu_values:
            model = KMeans(
                n_clusters=k, 
                init=initial_centroids, 
                metric='gini', 
                nu=nu, 
                n_init=1, 
                max_iter=300, 
                algorithm='lloyd',
            )
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred_silhouette = model.predict(X_train)
            silhouette_avg = silhouette_score(X_train, y_pred_silhouette)

            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_nu_silhouette = nu
            

            if not np.issubdtype(y_test.dtype, np.number):
                y_pred = align_clusters(y_test, y_pred)

            precision = precision_score(y_test, y_pred, average='weighted')

            if precision > best_precision:
                best_precision = precision
                best_nu = nu

        best_nu_per_fold.append(best_nu_silhouette)
        best_nu_values.append(best_nu)

        # Fit the model with the best nu 
        best_ginikmeans = KMeans(
            n_clusters=k, 
            init=initial_centroids, 
            metric='gini', 
            nu=best_nu, 
            n_init=1, 
            max_iter=300, 
            algorithm='lloyd'
        )
        best_ginikmeans.fit(X_train)
        # Fit the best nu with the Silhouette criteria
        best_ginikmeans_silhouette = KMeans(
            n_clusters=k, 
            init=initial_centroids, 
            metric='gini', 
            nu=best_nu_silhouette, 
            n_init=1, 
            max_iter=300, 
            algorithm='lloyd'
        )
        best_ginikmeans_silhouette.fit(X_train)
        precision_results["Gini_generalized_silhouette"].append(precision_score(y_test, best_ginikmeans_silhouette.predict(X_test), average='weighted'))
        recall_results["Gini_generalized_silhouette"].append(recall_score(y_test, best_ginikmeans_silhouette.predict(X_test), average='weighted'))
        precision_results['Gini_generalized'].append(best_precision)
        recall_results['Gini_generalized'].append(
            recall_score(y_test, best_ginikmeans.predict(X_test), average='weighted')
        )
        # Evaluation of other distances
        for distance in ['Gini', 'Euclidean', 'Manhattan', 'Minkowski', 'Cosine', 'Canberra', 'Hellinger', 'Jensen_Shannon', 'Pearson_Chi2', 'Vicis_Symmetric_1', 'Hassanat']:
            if distance == "gini":
                model = KMeans(
                n_clusters=k, 
                init=initial_centroids, 
                metric=distance.lower(), 
                n_init=1, 
                nu=2,
                max_iter=300, 
                algorithm='lloyd'
            )
            elif distance == "minkowski":
                model = KMeans(
                n_clusters=k, 
                init=initial_centroids, 
                metric=distance.lower(), 
                n_init=1, 
                p=3,
                max_iter=300, 
                algorithm='lloyd'
            )
            else:
                model = KMeans(
                    n_clusters=k, 
                    init=initial_centroids, 
                    metric=distance.lower(), 
                    n_init=1, 
                    max_iter=300, 
                    algorithm='lloyd'
                )
            model.fit(X_train)
            y_pred = model.predict(X_test)

            if not np.issubdtype(y_test.dtype, np.number):
                y_pred = align_clusters(y_test, y_pred)

            precision_results[distance].append(precision_score(y_test, y_pred, average='weighted'))
            recall_results[distance].append(recall_score(y_test, y_pred, average='weighted'))

        
    print(f"Dataset: {dataset_name}")
    for dist_name, result in best_results.items():
        precision_mean = np.mean(precision_results[dist_name])
        recall_mean = np.mean(recall_results[dist_name])

        if dist_name == 'Gini_generalized':
            best_results[dist_name]['Precision'] = precision_mean
            best_results[dist_name]['Recall'] = recall_mean
            best_results[dist_name]['Nu'] = np.mean(best_nu_values)
            print(f"{dist_name} - Meilleure précision: {precision_mean}, Rappel: {recall_mean} avec k={k}, nu={best_results[dist_name]['Nu']}")
        
        elif dist_name == 'Gini_generalized_silhouette':
            best_results[dist_name]['Precision'] = precision_mean
            best_results[dist_name]['Recall'] = recall_mean
            best_results[dist_name]['Nu'] = np.mean(best_nu_per_fold)
            print(f"{dist_name} - Meilleure précision: {precision_mean}, Rappel: {recall_mean} avec k={k}, nu={best_results[dist_name]['Nu']}")
        elif dist_name == 'Minkowski':
            best_results[dist_name]['Precision'] = precision_mean
            best_results[dist_name]['Recall'] = recall_mean
            print(f"{dist_name} - Meilleure précision: {precision_mean}, Rappel: {recall_mean} avec k={k}, p=3")
        else:
            best_results[dist_name]['Precision'] = precision_mean
            best_results[dist_name]['Recall'] = recall_mean
            print(f"{dist_name} - Meilleure précision: {precision_mean}, Rappel: {recall_mean} avec k={k}")

    print("\n")

    return best_results

def add_noise(X_train, noise_level):
    """
    Adds Gaussian noise to the dataset for testing robustness of clustering methods.
    - Inputs:
        X_train: Original dataset.
        noise_level: Standard deviation of Gaussian noise.
    - Returns:
        Dataset with added Gaussian noise.
    """
    noise = noise_level * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    return X_train + noise

# Charger les datasets
datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer(),
    'Sonar': fetch_openml(name='sonar', version=1, as_frame=False),
}
clusters_per_dataset = {
    'Iris': 3,
    'Wine': 3,
    'Breast Cancer': 2,
    'Digits': 10,
    'Sonar':2,
    'BankNote Authentication': 2,
    'Indian Liver Patient': 2,
    'Ionosphere': 2,
    'Australian':2 ,
    'Vehicle':3,
    'Heart': 4,
    'Glass': 6,
    'German':2,
    'Balance':3,
    'Haberman': 2,
    'Wholesale':2,
    'QSAR':3,
}

# Cas particuliers pour les datasets en .csv
def load_csv_datasets():
    """
    Loads and preprocesses custom datasets from CSV files.
    - Handles missing values, encodes categorical variables, and ensures numeric data formats.
    - Returns:
        A dictionary where keys are dataset names and values are (X, y) tuples.
    """
    csv_datasets = {
        'Australian': './australian.csv',
        'Heart': './heart.csv',
        'Glass': './glass.csv',
        'Balance':'./balance-scale.csv',
        'German':'./german_credit_data.csv',
        'Haberman': './haberman.csv',
        'Wholesale':'./Wholesale_customers_data.csv',
        'Vehicle': './vehicle.csv',
        'BankNote Authentication': './BankNote_Authentication.csv',
        'Indian Liver Patient': './indian_liver_patient.csv',
        'Ionosphere': './ionosphere_data.csv',
        'QSAR': './qsar.csv'
    }
    data = {}
    imputer = SimpleImputer(strategy='mean')
    for name, filepath in csv_datasets.items():
        df = pd.read_csv(filepath)

        if name == 'Indian Liver Patient':
            df = df.dropna()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            if 'Gender' in df.columns:
                df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
            X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
            y = pd.to_numeric(df.iloc[:, -1], errors='coerce').values
        elif name == 'Ionosphere':
            df = df.dropna()
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        elif name =="QSAR":
            X = df.iloc[:, 3:12].values
            y = df['Class'].values
            labelencoder_X_1 = LabelEncoder()
            y = labelencoder_X_1.fit_transform(y)
        elif name == 'BankNote Authentication':
            X = df.drop('class', axis=1).values
            y = df['class'].values
            y = LabelEncoder().fit_transform(y)
        elif name=="German":
            df = df.copy() 
            df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
            df['Checking account'] = df['Checking account'].fillna('unknown')
            label_encoders = {}
            for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            df = df.replace([np.inf, -np.inf], np.nan)
            X = df.drop(columns=['Credit amount']).values
            y = df['Credit amount'] > df['Credit amount'].median()
        elif name=='Australian':
            df.columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','Y']
            X=df.drop('Y',axis=1).values
            y=df['Y'].values
        elif name =="Liver":
            dataset = df.dropna()
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            encode_X = LabelEncoder()
            X[:,1] = encode_X.fit_transform(X[:,1])
        elif name == 'Vehicle':
            df['class'] = df['class'].replace({'car': 0, 'bus': 1, 'van': 2}).astype(int)
            X = df.drop('class', axis=1).values
            y = df['class'].values
            X = imputer.fit_transform(X)
        elif name=='Haberman':
            col_names = ['age', 'year', 'node', 'status']
            df.columns = col_names
            df = df.dropna()
            X = df.drop(columns='status').values 
            y = df['status'].values
        elif name=='Wholesale':
            X = df.drop('Channel', axis=1).values
            y = df['Channel']
        elif name == 'Balance':
            df['Class'] = LabelEncoder().fit_transform(df['Class'].tolist())
            y = df[['Class']].values
            X = df.drop(['Class'], axis = 1).values
        elif name=="Heart":
            X = df.drop(columns=['thal']).values
            y = df.thal
        else:
            df = df.dropna()
            X = df.drop(columns='Type').values 
            y = df['Type'].values 
            
        data[name] = (X, y)

    return data

csv_datasets = load_csv_datasets()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)
results = []
noise_levels = [0]

#Test standard datasets
with np.errstate(divide='ignore', invalid='ignore'):
    for name, dataset in datasets.items():
        for noise_level in noise_levels:
            X, y = dataset.data, dataset.target
            k = clusters_per_dataset[name]
            X_original = X.copy() 
            # Initialize Kmeans algorithm in order to keep the centroids
            kmeans_initial = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans_initial.fit(X_original)
            initial_centroids = kmeans_initial.cluster_centers_
            X_noisy = add_noise(X_original, noise_level)
            result = test(X_noisy, y, dataset_name=name, k=k, initial_centroids=initial_centroids)
            results.append((name, result))
warnings.filterwarnings("ignore", category=ConvergenceWarning)

with np.errstate(divide='ignore', invalid='ignore'):
    results = []
    noise_levels = [0, 0.05, 0.1]
    # Test CSV datasets
    for name, (X, y) in csv_datasets.items():
        for noise_level in noise_levels:
            X_noisy = add_noise(X, noise_level)
            k = clusters_per_dataset[name]
            X_original = X.copy() 
            # Initialize Kmeans algorithm in order to keep the centroids
            kmeans_initial = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans_initial.fit(X_original)
            initial_centroids = kmeans_initial.cluster_centers_

            X_noisy = add_noise(X_original, noise_level)
            result = test(X_noisy, y, dataset_name=name, k=k, initial_centroids=initial_centroids)
            results.append((name, result))
