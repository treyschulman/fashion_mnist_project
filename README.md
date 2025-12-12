# Fashion-MNIST Analysis: Clustering & Classification

This project performs an end-to-end machine learning analysis on the **Fashion-MNIST** dataset. It explores the data structure using unsupervised learning techniques and implements multiple supervised classification models to predict clothing categories. The goal is to compare the efficacy of linear vs. nonlinear models on high-dimensional image data.

## Dataset

  * **Source:** [Fashion-MNIST on Kaggle](https://www.google.com/search?q=https://www.kaggle.com/zalando-research/fashion-mnist)
  * **Description:** 70,000 grayscale images ($28 \times 28$ pixels)
  * **Classes:** 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
  * **Split:** 60,000 Training / 10,000 Testing

### 1\. Data Preprocessing

  * **Scaling:** Standardized pixel values (mean=0, variance=1).
  * **Dimensionality Reduction:** Applied Principal Component Analysis (PCA), retaining 95% of variance (reduced from 784 features to 256 components) for distance-based algorithms.
  * **Output:** Processed data saved as `.npz` for modular loading.

### 2\. Unsupervised Learning

  * **K-Means Clustering:** Optimal $k=10$ selected via Silhouette Analysis.
  * **Hierarchical Clustering:** Performed on a stratified subset ($N=20,000$) using Ward linkage; dendrogram analysis confirmed 10 distinct clusters.
  * **Key Findings:** Footwear and trousers formed distinct clusters, while upper-body garments (Coats, Shirts, Pullovers) showed significant overlap.

### 3\. Multi-class Classification

  * **Logistic Regression (Baseline):** Trained on PCA features ($Acc: 85.79\%$).
  * **Support Vector Machine (RBF Kernel):** Trained on PCA features ($Acc: 90.70\%$).
  * **Random Forest:** Trained on raw scaled pixels ($Acc: 88.44\%$).
  * **Gradient Boosting (XGBoost):** Trained on raw scaled pixels ($Acc: 91.22\%$).

## Results

Gradient Boosting achieved the highest accuracy, demonstrating that ensemble methods with nonlinear decision boundaries are best suited for this task. The confusion matrices highlighted that the primary source of error across all models was distinguishing between "Shirt", "T-shirt/top", and "Coat".

  * Python 3.x
  * `numpy`, `pandas`
  * `scikit-learn`
  * `matplotlib`, `seaborn`
  * `xgboost`
  * `scipy`
