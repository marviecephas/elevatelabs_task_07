# Task 7: Support Vector Machines (SVM) Classification

### Objective
The primary goal was to implement the **Support Vector Machine (SVM)** classifier for binary diagnosis, focusing on mandatory feature scaling, comparing different **Kernels** (Linear vs. RBF), and optimizing hyperparameters using **Grid Search**.

### Dataset & Preprocessing
* **Dataset:** Breast Cancer Wisconsin Dataset (Predicting 'diagnosis').
* **Preprocessing:** The data was cleaned (dropping ID/Null columns) and the target encoded.
* **Feature Scaling (Mandatory):** All predictor features were standardized using **`StandardScaler`**. SVM is a distance-based algorithm, meaning standardization is essential to prevent features with large scales from disproportionately influencing the **margin calculation**.

---

### Methodology: Kernel Comparison and Tuning

#### 1. Initial Kernel Evaluation
The model was first tested with two fundamental kernels:

| Kernel | Accuracy Score | Rationale |
| :--- | :--- | :--- |
| **Linear** | 0.9561 | Suitable for data that is linearly separable. |
| **RBF (Radial Basis Function)** | 0.9825 | **Preferred Kernel.** Essential for data that is not linearly separable, implicitly mapping data to a high dimension (Kernel Trick). |

#### 2. Hyperparameter Tuning using Grid Search
The **RBF Kernel** requires tuning of two parameters that control the margin and complexity:
* **$C$ (Regularization):** Controls the penalty for misclassification.
* **$\gamma$ (Gamma):** Controls the influence of a single training example on the decision boundary.

**`GridSearchCV`** was implemented with **5-Fold Cross-Validation** (CV) to test multiple combinations of $C$ and $\gamma$ and ensure the best parameters are robust across different data samples. 


#### **Optimal Parameters Found**
* **Best Parameters ($C$ and $\gamma$):** **{'C': 100, 'gamma': 0.001}**
* **Final Tuned Accuracy:** **0.9825**

---

### Conceptual Insights

#### 1. Core Mechanism
* **Margin Maximization:** SVM's goal is to find the **optimal hyperplane** that creates the largest possible separation (the **margin**) between the classes. 
* **Support Vectors:** Only the data points lying closest to the margin (the **Support Vectors**) define the decision boundary.

#### 2. Handling Non-Linearity
When data is not linearly separable, the **Kernel Trick** is used. The RBF kernel implicitly transforms the data into a higher-dimensional space where a linear hyperplane can be found.

#### 3. Overfitting Control
Overfitting in SVM is managed by tuning the **$C$ parameter** (penalty control) and the **$\gamma$ parameter** (influence control). A high $C$ or a high $\gamma$ increases complexity and the risk of **overfitting**.
