import numpy as np
import pandas as pd
from scipy.spatial.distance import  canberra, jensenshannon
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
import warnings
import scipy.stats as ss
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


class GiniDistance:
    """
    Computes the Gini-based distance for a dataset.
    - Uses ranks of the data to calculate Gini distances.
    - Supports a customizable Gini parameter (`gini_param`).
    """
    def __init__(self, X, gini_param=2):
        self.X = X
        self.gini_param = gini_param

    def _rank(self, X):
        """
        Compute ranks for the given data array.
        Used to calculate the cumulative ranks for Gini distance.
        """
        ranks = np.apply_along_axis(ss.rankdata, 0, X)
        return X.shape[0] - ranks + 1

    def compute_gini_ranks(self, X):
        """
        Compute cumulative ranks for training and test data.
        Adjusts ranks based on the Gini parameter.
        """
        X_cat = np.concatenate((self.X, X), axis=0)
        ranks = (self._rank(X_cat) / X_cat.shape[0] * self.X.shape[0]) ** (self.gini_param - 1)
        return ranks[:self.X.shape[0]], ranks[self.X.shape[0]:]

    def gini_distance(self, x, Y, decum_rank_x, decum_ranks_Y):
        """
        Calculate the Gini distance between a single point `x` and a set of points `Y`.
        Combines rank differences with feature differences to compute the distance.
        """
        distance = -np.sum((x - Y) * (decum_rank_x - decum_ranks_Y), axis=1)
        return distance

    def compute_distances(self, X):
        """
        Compute the Gini distance matrix for test data `X` relative to training data.
        Returns a precomputed distance matrix.
        """
        ranks_train, ranks_test = self.compute_gini_ranks(X)
        distances = np.zeros((X.shape[0], self.X.shape[0]))
        
        for i, x in enumerate(X):
            distances[i, :] = self.gini_distance(x, self.X, ranks_test[i], ranks_train)
        return distances

def evaluate_knn(train_distances, test_distances, y_train, y_test, k):
    """
    Train and evaluate a KNN classifier using precomputed distance matrices.
    Returns precision and recall scores.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    knn.fit(train_distances, y_train)
    y_pred = knn.predict(test_distances)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return precision, recall


def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two probability distributions.
    Suitable for comparing normalized data.
    """
    p = np.asarray(np.abs(p))
    q = np.asarray(np.abs(q))
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def pearson_chi2(x, y):
    """
    Compute the Pearson Chi-Squared distance between two arrays.
    Useful for categorical or frequency-based data.
    """
    return np.sum((x - y) ** 2 / (x + y))

def vicis_symmetric_1(x, y):
    """
    Compute the Vicis Symmetric 1 distance between two arrays.
    Handles edge cases when the minimum value is zero.
    """
    total = 0
    for xi, yi in zip(x, y):
        if min(xi, yi) != 0:
            total += (xi - yi) ** 2 / (min(xi, yi) ** 2)
        else:
            total += (xi - yi) ** 2  # Handle the case when min(xi, yi) == 0
    return total


def HasD(x, y):
    """
    Compute the Hassanat distance (HasD) between two arrays.
    A distance metric that considers the ratio of minimum and maximum values.
    """
    total = 0
    for xi, yi in zip(x, y):
        min_value = min(xi, yi)
        max_value = max(xi, yi)
        total += 1 
        if min_value >= 0:
            total -= (1 + min_value) / (1 + max_value)
        else:
            total -= 1 / (1 + max_value + abs(min_value))
    return total


def test(X, y, dataset_name):
    """
    Test KNN with various distance metrics on a given dataset.
    - Performs K-fold cross-validation.
    - Optimizes hyperparameters (e.g., k, Gini parameter).
    - Evaluates precision and recall for each distance metric.
    """
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(kf.split(X))
    best_results = {
        'Gini_generalized': {'k': 0, 'Precision': 0, 'Recall': 0, 'Nu': 0},
        'Gini': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Euclidean': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Manhattan': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Minkowski': {'k': 0, 'Precision': 0, 'Recall': 0, 'p': 3},
        'Cosine': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Canberra': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Hellinger': {'k': 0, 'Precision': 0, 'Recall': 0},
        'JensenShannon': {'k': 0, 'Precision': 0, 'Recall': 0},
        'PearsonChi2': {'k': 0, 'Precision': 0, 'Recall': 0},
        'VicisSymmetric1': {'k': 0, 'Precision': 0, 'Recall': 0},
        'Hassanat': {'k': 0, 'Precision': 0, 'Recall': 0},
    }
    #Loop over the k neighbors
    for k in range(1, 11): 
        #Grid search for the nu parameter of Gini distance
        for nu in np.arange(1.1, 5.1, 0.01):
            fold_precisions = []
            fold_recalls = []

            for train_index, test_index in splits:
                #Split dataset into training and testing
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                #Compute Gini distance
                gini_calculator = GiniDistance(X_train, gini_param=nu)
                train_distances_gini = gini_calculator.compute_distances(X_train)
                test_distances_gini = gini_calculator.compute_distances(X_test)

                train_distances_gini = np.nan_to_num(train_distances_gini, nan=100, posinf=100, neginf=-100)
                test_distances_gini = np.nan_to_num(test_distances_gini, nan=100, posinf=100, neginf=-100)

                #Compute scores
                precision, recall = evaluate_knn(train_distances_gini, test_distances_gini, y_train, y_test, k)
                
                fold_precisions.append(precision)
                fold_recalls.append(recall)

            mean_precision = np.mean(fold_precisions)
            mean_recall = np.mean(fold_recalls)

            if mean_precision > best_results['Gini_generalized']['Precision']:
                best_results['Gini_generalized'] = {
                    'k': k,
                    'Precision': mean_precision,
                    'Recall': mean_recall,
                    'Nu': nu
                }
    #Other distances
    for distance_name, metric in zip(['Euclidean', 'Manhattan', 'Minkowski', 'Cosine', 'Gini', 'Canberra', 'Hellinger', 
                                      'JensenShannon', 'PearsonChi2', 'VicisSymmetric1', 'Hassanat'], 
                                     ['euclidean', 'manhattan', 'minkowski', 'cosine', 'gini', canberra, hellinger_distance,
                                      jensenshannon, pearson_chi2, vicis_symmetric_1, HasD]):
        for k in range(1, 11):
            fold_precisions = []
            fold_recalls = []

            for train_index, test_index in splits:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if (metric ==hellinger_distance) or (metric==pearson_chi2):
                    X_train = np.abs(X_train)
                    X_test = np.abs(X_test)
                if metric == 'minkowski':
                    train_distances = pairwise_distances(X_train, X_train, metric=metric, p=3)
                    test_distances = pairwise_distances(X_test, X_train, metric=metric, p=3)
                elif metric == 'gini':
                    gini_calculator = GiniDistance(X_train, gini_param=2)
                    train_distances = gini_calculator.compute_distances(X_train)
                    test_distances = gini_calculator.compute_distances(X_test)
                elif callable(metric):
                    train_distances = pairwise_distances(X_train, X_train, metric=metric)
                    test_distances = pairwise_distances(X_test, X_train, metric=metric)
                else:
                    train_distances = pairwise_distances(X_train, X_train, metric=metric)
                    test_distances = pairwise_distances(X_test, X_train, metric=metric)

                # Replace NaN
                train_distances = np.nan_to_num(train_distances, nan=100, posinf=100, neginf=-100)
                test_distances = np.nan_to_num(test_distances, nan=100, posinf=100, neginf=-100)

                precision, recall = evaluate_knn(train_distances, test_distances, y_train, y_test, k)
                
                fold_precisions.append(precision)
                fold_recalls.append(recall)

            mean_precision = np.mean(fold_precisions)
            mean_recall = np.mean(fold_recalls)

            if mean_precision > best_results[distance_name]['Precision']:
                best_results[distance_name] = {
                    'k': k,
                    'Precision': mean_precision,
                    'Recall': mean_recall
                }

    print(f"Dataset: {dataset_name}")
    for dist_name, result in best_results.items():
        if dist_name == 'Gini_generalized':
            print(f"{dist_name} - Best precision: {result['Precision']}, Recall: {result['Recall']} with k={result['k']}, nu={result['Nu']}")
        elif dist_name == 'Minkowski':
            print(f"{dist_name} -  Best precision: {result['Precision']}, Recall: {result['Recall']} with k={result['k']}, p=3")
        else:
            print(f"{dist_name} - Best precision: {result['Precision']}, Recall: {result['Recall']} with k={result['k']}")
    print("\n")

    return best_results



# Ajouter du bruit gaussien
def add_noise(X_train, noise_level):
    """
    Add Gaussian noise to the data to simulate noisy conditions.
    """
    noise = noise_level * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    return X_train + noise

# Standard dataset
datasets = {
    'Iris': load_iris(),
    'Wine': load_wine(),
    'Breast Cancer': load_breast_cancer(),
    'Sonar': fetch_openml(name='sonar', version=1, as_frame=False),
}

def load_csv_datasets():
    """
    Load custom datasets from CSV files and preprocess them.
    Handles categorical variables, missing data, and specific dataset requirements.
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
            df['class'] = df['class'].replace({'car': 0, 'bus': 1, 'van': 2})
            X = df.drop('class', axis=1).values
            y = df['class'].values
            X = imputer.fit_transform(X)

        elif name == 'Haberman':
            col_names = ['age', 'year', 'node', 'status']
            df.columns = col_names
            X = df.drop('status', axis=1).values
            y = df['status'].values
            y = LabelEncoder().fit_transform(y)

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
            df['Type'] = LabelEncoder().fit_transform(df['Type'])
            X = df.drop(columns='Type').values
            y = df['Type'].values

        data[name] = (X, y)

    return data

csv_datasets = load_csv_datasets()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)
results = []
# Different level of noise
noise_levels = [0, 0.05, 0.10]

# Test standard datasets
for name, dataset in datasets.items():
    for noise_level in noise_levels:
        X, y = dataset.data, dataset.target
        X_noisy = add_noise(X, noise_level) # Add noise to the data to evaluate robustness of the model
        result = test(X_noisy, y, dataset_name=name)
        results.append((name, result))

# Test CSV datasets
for name, (X, y) in csv_datasets.items():
    for noise_level in noise_levels:
        X_noisy = add_noise(X, noise_level)
        result = test(X_noisy, y, dataset_name=name)
        results.append((name, result))

# Save results in a CSV file
results_df = pd.DataFrame(results, columns=['Dataset', 'Best Results'])
results_df.to_csv('best_distance_precision_recall_results.csv', index=False)
