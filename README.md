
# spgdeslib

## Description
The `spgdeslib` is a Python library that implements the Self-generating Prototype Dynamic Selection Ensemble (SGPDES) algorithm. This package provides functionalities for creating and training a dynamic ensemble of classifiers with self-generating prototype selection.

## Installation

To install `spgdeslib` directly from the GitHub repository, follow the instructions below:

### Requirements
Ensure you have Python 3.6 or higher installed in your environment. You can check the Python version using the following command:

```bash
python --version
```

### Installation Steps

1. **Install via pip**

   To install the package directly from GitHub, you can use the following `pip` command:

   ```bash
   pip install --force-reinstall --no-cache-dir git+https://github.com/manastarla/spgdeslib.git
   ```

   This command ensures that the repository is cloned again and that no cached version is used.

2. **Installation in Google Colab**

   If you are using Google Colab, you can run the command above in a code cell:

   ```python
   !pip install --force-reinstall --no-cache-dir git+https://github.com/manastarla/spgdeslib.git
   ```

## Usage

After installation, you can import and use `spgdeslib` in your Python projects. Here is a basic example of how to use the package:

### Example Usage

```python
# Import the necessary class from the spgdeslib package
from spgdes.poolgenerator import PoolGenerator

# Example usage of the PoolGenerator class
n_classifiers = 10
X_train = ...  # your training dataset
y_train = ...  # your training labels

# Initialize the pool generator
pool_gen = PoolGenerator(n_classifiers=n_classifiers)

# Generate and train the pool of classifiers
pool = pool_gen.generate_pool(X_train, y_train)

# Now you can use 'pool' for predictions, evaluations, etc.
```

## Contribution

If you wish to contribute to the development of `spgdeslib`, follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes.
4. Submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contact

For any questions or issues, please open an issue in the GitHub repository or contact the author at manastarla@hotmail.com.
