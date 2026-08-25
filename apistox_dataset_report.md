# **Machine Learning Model Analysis & Preprocessing Report**

## **1. Data Preprocessing Overview**
Before model training, the dataset underwent several preprocessing steps to ensure optimal performance. Below is a detailed breakdown of the transformations applied:

### **1.1 Data Cleaning**
- **Columns Cast to Categorical:**
  - `CID`, `source`, and `toxicity_type` were explicitly cast as categorical variables to ensure proper handling in encoding and modeling.

### **1.2 Feature Encoding**
To convert categorical variables into numerical representations for model compatibility, the following encoding methods were applied:
- **One-Hot Encoding:**
  - Applied to `source` and `toxicity_type` to avoid ordinal assumptions and allow the model to interpret each category independently.
- **Binary Encoding:**
  - Applied to `name`, `CID`, `CAS`, and `SMILES` to reduce dimensionality compared to one-hot encoding while still capturing categorical relationships.

### **1.3 Handling Class Imbalance**
The dataset exhibited class imbalance, which was addressed using **resampling techniques** to improve minority class representation:
- **SMOTE (Synthetic Minority Over-sampling Technique):**
  - Generates synthetic samples for the minority class to balance class distribution.
- **SMOTETomek:**
  - Combines SMOTE with Tomek Links (a method to clean overlapping samples between classes) for better class separation.
- **ADASYN (Adaptive Synthetic Sampling):**
  - Similar to SMOTE but focuses on generating samples near the decision boundary for improved learning.

### **1.4 Feature Scaling**
- **RobustScaler** was applied to all numerical features before model training.
  - **Why RobustScaler?**
    - Scales features by subtracting the median and dividing by the interquartile range (IQR), making it **robust to outliers** (unlike StandardScaler or MinMaxScaler).
    - Ensures that features with large value ranges (e.g., `year`, `fungicide`, `insecticide`) do not dominate the model due to scale differences.

### **1.5 Handling Skewed Features**
- **Log Transformation (or other skew correction methods) was applied to:**
  - `year`, `fungicide`, `insecticide`, and `other_agrochemical`
  - **Why?**
    - These features likely had a **non-normal distribution**, which can negatively impact models that assume normally distributed data (e.g., linear models, though tree-based models are less sensitive).
    - Helps improve model convergence and performance.

---

## **2. Model Training & Hyperparameter Tuning**
Three **ensemble-based models** were trained and evaluated:
1. **Random Forest**
2. **XGBoost (Extreme Gradient Boosting)**
3. **LightGBM (Light Gradient Boosting Machine)**

### **2.1 Hyperparameters Used**
Each model was configured with optimized hyperparameters to prevent overfitting and improve generalization:

| **Model**      | **Key Hyperparameters** |
|---------------|------------------------|
| **Random Forest** | `n_estimators=300`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `bootstrap=True`, `criterion='gini'` |
| **XGBoost**    | `n_estimators=300`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.1`, `gamma=0.1`, `reg_alpha=0.1`, `reg_lambda=1.0` |
| **LightGBM**   | `n_estimators=300`, `max_depth=5`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.05`, `reg_alpha=0.1`, `reg_lambda=0.1` |

### **2.2 Evaluation Metrics**
Models were evaluated using:
- **F1-Score** (Harmonic mean of precision and recall, good for imbalanced data)
- **ROC-AUC** (Area under the ROC curve, measures class separation ability)
- **Recall** (Sensitivity, proportion of actual positives correctly identified)

---

## **3. Model Performance Summary**
The top-performing models after evaluation were:

| **Model** | **F1-Score** | **ROC-AUC** | **Recall** | **False Positives** |
|-----------|-------------|------------|-----------|---------------------|
| `RobustScaler-random_forest-SMOTE` | 0.745 | 0.798 | 0.603 | 1 |
| **`RobustScaler-random_forest-SMOTETomek`** | **0.765** | **0.810** | **0.619** | **0** |
| **`RobustScaler-lightgbm-SMOTETomek`** | **0.765** | **0.810** | **0.619** | **0** |

### **Key Takeaways:**
- **`RobustScaler-random_forest-SMOTE`** performed slightly worse, with **1 false positive** and lower metrics.
- **`RobustScaler-random_forest-SMOTETomek` and `RobustScaler-lightgbm-SMOTETomek`** tied for the best performance:
  - **Highest F1 (0.765), ROC-AUC (0.810), and Recall (0.619).**
  - **Zero false positives**, meaning no negative cases were misclassified as positive.

---

## **4. Final Model Selection: `RobustScaler-random_forest-SMOTETomek`**
The user selected **`RobustScaler-random_forest-SMOTETomek`** as the final model. Below are its **pros and cons**:

### **Pros:**
✅ **High Performance:**
   - Achieves the **best F1-score (0.765), ROC-AUC (0.810), and Recall (0.619)** among the evaluated models.
   - **No false positives**, making it reliable for applications where avoiding false alarms is critical.

✅ **Handles Imbalanced Data Well:**
   - **SMOTETomek** effectively balances the dataset while cleaning noisy samples, improving minority class detection.

✅ **Robust to Outliers & Feature Scales:**
   - **RobustScaler** ensures that outliers do not distort the model’s learning process.

✅ **Interpretability:**
   - Random Forest provides **feature importance scores**, allowing for explainability of predictions.

✅ **Generalization:**
   - Hyperparameters (`max_depth=10`, `min_samples_split=5`) prevent overfitting while maintaining model flexibility.

### **Cons:**
❌ **Computationally Expensive:**
   - Random Forest trains multiple decision trees, which can be **slower than LightGBM** for very large datasets.

❌ **Less Efficient for High-Dimensional Data:**
   - If the dataset grows significantly in features, Random Forest may become **less scalable** compared to gradient boosting methods.

❌ **Memory Usage:**
   - Stores all decision trees in memory, which can be **resource-intensive** for very deep forests.

---

## **5. Conclusion**
The preprocessing pipeline included **categorical encoding, robust scaling, skew correction, and class imbalance handling** to prepare the data for modeling. Among the evaluated models, **`RobustScaler-random_forest-SMOTETomek`** was selected as the final model due to its **superior performance, reliability (zero false positives), and interpretability**. While it may not be the fastest or most memory-efficient option, its **strong predictive power and explainability** make it a robust choice for this task.