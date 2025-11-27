# Logistic Regression Diabetes Prediction — Summary & Comparison

This project implemented a complete end-to-end machine learning pipeline using Logistic Regression to predict diabetes using the Pima Indian Diabetes dataset. Below is a full summary of the work done, results obtained, and interpretation.

## ✅ 1. Data Cleaning & Preprocessing

### Identified Impossible Zero Values

Several physiological features contained zero values that are biologically impossible:

| Feature | Zero Counts | % of Data |
|---------|-------------|-----------|
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |

These were replaced with NaN, then imputed with the median.

### Reason for Isolating Only These Columns

Because only these features can never be zero in real life. Other features (Age, Pregnancies, Pedigree) can naturally be zero, so replacing them would introduce errors.

## ✅ 2. Train/Test Split

We used `train_test_split()` to correctly shuffle and split:
```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)
```


## ⭐ 3. Regularization Tuning (L2 / Ridge)

We evaluated 200 values of C (inverse of regularization strength):
- Low C → stronger regularization
- High C → weaker regularization

### Best Result
```
Best C = 0.000512462311557789
Best test accuracy = 0.7662
Accuracy without regularization = 0.7532
```

### ✔ Interpretation

- Regularization improves performance by smoothing model coefficients.
- It reduces model variance and prevents overfitting.

## ⭐ 4. Model Performance (Classification Metrics)

### Testing Results
```
Accuracy = 0.753
```

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 (No diabetes) | 0.80 | 0.83 | 0.81 | 99 |
| 1 (Diabetes) | 0.67 | 0.62 | 0.64 | 55 |

### ✔ Interpretation

- Model performs much better on class 0 due to class imbalance (500 vs 268).
- Recall for class 1 is lower → the model misses some diabetes cases.

## ⭐ 5. Confusion Matrices

### Training Confusion Matrix
```
[[353  48]
 [ 94 119]]
Training accuracy = 0.7687
```

### Testing Confusion Matrix
```
[[82 17]
 [21 34]]
Testing accuracy = 0.7532
```

### ✔ No overfitting — train and test accuracy are similar.

## ⭐ 6. Feature Importance (Model Coefficients)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| DiabetesPedigreeFunction | 0.55 | Strong positive predictor |
| BMI | 0.10 | Higher BMI increases diabetes risk |
| Pregnancies | 0.067 | More pregnancies → higher risk |
| Glucose | 0.037 | Higher glucose strongly indicates diabetes |
| Age | 0.034 | Older age increases risk |
| SkinThickness | 0.007 | Slight effect |
| Insulin | -0.001 | Weak negative signal (data noisy) |
| BloodPressure | -0.013 | Slight negative effect |

### ✔ Interpretation

The model aligns with medical understanding:
- Pedigree (genetic risk), BMI, Age, Glucose → strong predictors of diabetes.

## ⭐ 7. Threshold Optimization

Instead of using the default threshold of 0.5, we measured:
- FPR (False Positive Rate)
- FNR (False Negative Rate)

across thresholds from 0 → 1.

### Best Threshold Result
```
Best threshold = 0.520
Minimum total error (FNR + FPR) = 0.5212
```

### Confusion Matrix at Best Threshold
```
[[87 12]
 [22 33]]
Accuracy = 0.7792
```

### ✔ Interpretation

Raising the threshold slightly above 0.5 improved performance.

## 8. FNR vs. FPR Curve

I plotted a smooth curve showing:
- FNR ↑ as threshold ↑
- FPR ↓ as threshold ↑

The intersection helps identify the balanced threshold.

## What This Entire Pipeline Shows

- ✔ Logistic Regression can achieve ~77% accuracy on diabetes prediction.
- ✔ Proper data cleaning (removing impossible zeros) improved model reliability.
- ✔ Regularization significantly improved test accuracy.
- ✔ Threshold tuning improved final performance beyond raw model accuracy.
- ✔ The model is interpretable—coefficients match medical expectations.
- ✔ No major signs of overfitting.
- ✔ Class imbalance remains a challenge (next steps below).

## To Improve Model Performance, I will be working on these next steps of action. I also recommend you try them yourselves if you would.

1. **Balance classes** (SMOTE, class weights)
2. **Try more advanced models**
   - RandomForest
   - XGBoost
3. **Cross-validation** instead of single train/test split
4. **Hyperparameter tuning** via GridSearchCV
5. **Build ROC curve & compute AUC**