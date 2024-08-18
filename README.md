
# spgdes

## Description
The `spgdes` is a Python library that implements the Self-generating Prototype Dynamic Selection Ensemble (SGPDES) and HSGP algorithms. This package provides functionalities for creating and training a dynamic ensemble of classifiers with self-generating prototype selection.

## Installation

To install `spgdes` directly from the GitHub repository, follow the instructions below:

### Requirements
Ensure you have Python 3.6 or higher installed in your environment. You can check the Python version using the following command:

```bash
python --version
```

### Installation Steps 

1. **Install via pip**

   To install the package directly from GitHub, you can use the following `pip` command:

   ```bash
   pip install --force-reinstall --no-cache-dir git+https://github.com/manastarla/sgpdes.git
   ```

   This command ensures that the repository is cloned again and that no cached version is used.

2. **Installation in Google Colab**

   If you are using Google Colab, you can run the command above in a code cell:

   ```python
   # If you want to install a specific branch, you can do that as well
   !pip install git+https://github.com/manastarla/sgpdes.git
   ```

3. **Install DESlib for DES method comparison**

   To install `DESlib` for comparison with the DES methods:

   ```python
   !pip install git+https://github.com/scikit-learn-contrib/DESlib
   ```

## Usage

After installation, you can import and use `spgdes` in your Python projects. Here is a basic example of how to use the package:

### Example Usage

```python
# Importing the SGPDES technique
from sgpdes.spgdes import SGPDES

# Perceptron PoolGenerator
from util.poolgenerator import PoolGenerator

# DES techniques from DESlib
from deslib.des import KNORAE, KNORAU, DESP, METADES
import warnings
import requests
import zipfile
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Function to download and unzip the dataset
def download_and_unzip(url, local_filename, extract_to):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as file:
            file.write(response.content)
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise Exception(f"Failed to download file from {url}")
    print(f"Downloaded and extracted: {local_filename}")

# Function to load dataset
def load_data(name):
    data_path = f"{name}/{name}.dat"
    data = pd.read_csv(data_path, comment='@', header=None)
    X = data.iloc[:, :-1].astype(float)
    y = pd.Categorical(data.iloc[:, -1].astype(str).str.strip(), categories=["positive", "negative"], ordered=True).codes
    return X, pd.Series(y)

# Function to evaluate each method
def evaluate_method(ctl, X_train, X_test, y_train, y_test, method_name, fold, ir):
    ctl.fit(X_train, y_train)
    y_pred = ctl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    print(f"Dataset Fold {fold}: Accuracy {method_name} = {accuracy:.3f}, F1 Score {method_name} = {f1:.3f}, IR = {ir:.3f}")
    return accuracy, f1

# Function to process each dataset
def process_dataset(dataset, methods):
    name = dataset["name"]
    download_and_unzip(dataset["url"], f"{name}.zip", name)
    X, y = load_data(name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {method: {"accuracies": [], "f1_scores": []} for method in methods}
    imbalance_ratios, reduction_rates_sgpdes = [], []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ir = y_train.value_counts().max() / y_train.value_counts().min()
        imbalance_ratios.append(ir)

        pool_generator = PoolGenerator(n_classifier=10)
        pool_classifiers = pool_generator.PoolGeneration(X_train, X_test, y_train, y_test)

        for method_name, ctl in methods.items():
            accuracy, f1 = evaluate_method(ctl(pool_classifiers), X_train, X_test, y_train, y_test, method_name, fold, ir)
            results[method_name]["accuracies"].append(accuracy)
            results[method_name]["f1_scores"].append(f1)

            if "SGPDES" in method_name:
                _, reduction_rate = ctl(pool_classifiers).fit(X_train, y_train)
                reduction_rates_sgpdes.append(reduction_rate)

    summary = []
    for method, metrics in results.items():
        summary.append({
            "Dataset": name,
            "Method": method,
            "Mean Accuracy": np.mean(metrics["accuracies"]),
            "Std Accuracy": np.std(metrics["accuracies"]),
            "Mean F1 Score": np.mean(metrics["f1_scores"]),
            "Std F1 Score": np.std(metrics["f1_scores"]),
            "Reduction Rate DSEL": np.mean(reduction_rates_sgpdes) if "SGPDES" in method else None
        })

    return summary, np.mean(imbalance_ratios)

# Dictionary of methods for dynamic instantiation
methods = {
    "METADES": lambda pool: METADES(pool),
    "KNORAE": lambda pool: KNORAE(pool),
    "KNORAU": lambda pool: KNORAU(pool),
    "DESP": lambda pool: DESP(pool),
    "SGPDES KNN": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDKNN", CONSENSUSTH=101, resultprint=False),
    "SGPDES RF": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDRF", CONSENSUSTH=101, resultprint=False),
    "SGPDES SVM": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDSVM", CONSENSUSTH=101, resultprint=False),
    "SGPDES XGB": lambda pool: SGPDES(WMA=25, ESD=0.001, EL=0.9, KI=1, pool_classifiers=pool, DESNumbNN=7, Selector_Mode="MODELBASEDXGB", CONSENSUSTH=101, resultprint=False)
}

# List of datasets to process
datasets = [
    {"name": "glass1", "url": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/glass1.zip"}
]

# Process all datasets and save results
all_results = []
for dataset in datasets:
    summary, mean_ir = process_dataset(dataset, methods)
    all_results.extend(summary)
    print(f"Dataset: {dataset['name']}, Mean IR of 5 folds = {mean_ir:.3f}")

# Save results to CSV
results_df = pd.DataFrame(all_results)
#results_df.to_csv("results.csv", index=False)

print("results_df")
````

## Contribution

If you wish to contribute to the development of `spgdes`, follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes.
4. Submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact

For any questions or issues, please open an issue in the GitHub repository or contact the author at manastarla@hotmail.com.

## Usage of HSGP

You can use the `HSGP` function from the `ps.hsgp` module to perform prototype selection. Below is an example of how to use the `HSGP` function with the Breast Cancer dataset:

```python
from ps.hsgp import HSGP
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load a sample dataset (Breast Cancer Dataset)
data = load_breast_cancer()
X = data.data
y = data.target

# Combine the features and target into a single array for the HSGP algorithm
TR = np.column_stack((X, y))

# Define the parameters for the HSGP algorithm
WMA = 3  # Window Moving Average
ESD = 0.001  # Entropy Standard Deviation
EL = 0.1  # Entropy Level for instance selection
KI = 1  # Number of nearest neighbors
max_iter = 1000  # Maximum number of iterations

# Run the HSGP algorithm
R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes = HSGP(TR, WMA, ESD, EL, KI, max_iter)

# Output the results
print(f"Original Training Accuracy: {accuracy_TR:.2f}%")
print(f"Reduced Training Accuracy: {accuracy_R:.2f}%")
print(f"Reduction Rate: {100-reduction_rate:.2f}%")
```

This example demonstrates how to load a dataset, configure the HSGP algorithm, and then execute it to obtain prototypes and evaluate the reduction in training data size.
