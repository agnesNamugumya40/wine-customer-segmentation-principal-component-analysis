# Principal Component Analysis (PCA) — Wine Customer Segmentation

A machine learning project that uses **Principal Component Analysis (PCA)** to reduce 13 chemical features of wines down to **2 principal components**, then trains a Logistic Regression model to predict which customer segment a wine belongs to.

## What is PCA?

PCA is an **unsupervised** dimensionality reduction technique. It finds new axes, called **principal components**, along which the data varies the most. The first component captures the most variance, the second captures the most remaining variance, and so on.

Keeping only the first few components:

- reduces the number of features, making models faster and simpler
- removes redundancy between correlated features
- makes it possible to **visualise** high-dimensional data in 2D

PCA does not use the target labels; it only looks at the features.

## Dataset

**File:** `Wine.csv`

- 178 wines, each described by 13 chemical properties
- Target: `Customer_Segment` — 3 segments (59, 71 and 48 wines)

| Features |
|---|
| `Alcohol`, `Malic_Acid`, `Ash`, `Ash_Alcanity`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Nonflavanoid_Phenols`, `Proanthocyanins`, `Color_Intensity`, `Hue`, `OD280`, `Proline` |

## Workflow

The notebook (`principal_component_analysis.ipynb`) follows these steps:

1. **Import libraries** — NumPy, Matplotlib, Pandas
2. **Load dataset** — 13 features `X`, target `y` (`Customer_Segment`)
3. **Train/test split** — 80% training (142 wines), 20% test (36 wines), `random_state=0`
4. **Feature scaling** — StandardScaler, which is required before PCA so that features with large values (e.g. `Proline`) do not dominate
5. **Apply PCA** — reduces the 13 features to 2 principal components
6. **Model training** — Logistic Regression on the 2 components
7. **Evaluation** — confusion matrix and accuracy score
8. **Visualisation** — decision regions for the training and test sets, plotted on PC1 vs. PC2

## Requirements

```
numpy
matplotlib
pandas
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib pandas scikit-learn
```

## Usage

Open the notebook in Jupyter or Google Colab and run all cells in order:

```bash
jupyter notebook principal_component_analysis.ipynb
```

## Model Configuration

```python
PCA(n_components = 2)
LogisticRegression(random_state = 0)
```

## Results

**Variance explained by the 2 components:**

| Component | Variance explained |
|---|---:|
| PC1 | 36.9% |
| PC2 | 19.3% |
| **Total** | **56.2%** |

**Confusion matrix (test set, 36 wines):**

| | Predicted 1 | Predicted 2 | Predicted 3 |
|---|---:|---:|---:|
| **Actual 1** | 14 | 0 | 0 |
| **Actual 2** | 1 | 15 | 0 |
| **Actual 3** | 0 | 0 | 6 |

**Accuracy: 97.2%** (35 of 36 wines classified correctly)

Even though the 2 components keep only about 56% of the original variance, they are enough to separate the three customer segments almost perfectly, with only one wine misclassified. The decision-region plots show the three segments forming clearly separated groups in the PC1–PC2 space.

## Key Concepts

- **Dimensionality reduction**: representing data with fewer features while keeping as much useful information as possible.
- **Principal component**: a new feature made from a weighted combination of the original features, chosen to capture as much variance as possible.
- **Explained variance ratio**: the share of the data's total variance captured by each component.
- **Unsupervised**: PCA ignores the target labels when finding components (unlike LDA).
- **Feature scaling**: essential for PCA, since it is based on variance and would otherwise favour large-scale features.
