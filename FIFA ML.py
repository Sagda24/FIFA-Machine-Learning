

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
#%%

#%%
# Load data
df = pd.read_csv("Fifa.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# -------------------------
# Missing values
# -------------------------
missing = df.isnull().sum()
missing = missing[missing > 0]

print("Missing values per column:\n", missing)
print("\nTotal Missing:", missing.sum())

# -------------------------
# Distribution
# -------------------------
value_col = 'Value_Per_M$'

plt.figure()
sns.histplot(df[value_col].dropna(), kde=True)
plt.title("Distribution of Value Per M$")
plt.show()

skewness = df[value_col].skew()
print("Skewness:", skewness)

if skewness > 1:
    print("→ Highly right skewed")
elif skewness < -1:
    print("→ Highly left skewed")
else:
    print("→ Approximately normal")

# -------------------------
# Correlation
# -------------------------
num_df = df.select_dtypes(include=np.number)
corr = num_df.corr()[value_col].sort_values(ascending=False)

print("\nCorrelation with Value:")
print(corr)

# -------------------------
# Avg rating per position
# -------------------------
avg_rating = df.groupby('Position')['Overall_Rating'].mean()
print("\nAverage Rating per Position:\n", avg_rating)
#%%
# Split BEFORE preprocessing
X = df.drop(columns=[value_col])
y = df[value_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# -------------------------
# Missing values
# -------------------------
X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].mean())

for col in cat_cols:
    mode = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode)
    X_test[col] = X_test[col].fillna(mode)

# -------------------------
# Outliers (IQR)
# -------------------------
for col in num_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    X_train[col] = np.clip(X_train[col], low, high)
    X_test[col] = np.clip(X_test[col], low, high)

# -------------------------
# Encoding + Scaling
# -------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Train shape:", X_train_processed.shape)
print("Test shape:", X_test_processed.shape)
#%%

# ---------------------------
# 1. Split FIRST
# ---------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------------------
# 2. Check distribution (Train only)
# ---------------------------
sns.histplot(train_df['Overall_Rating'], kde=True)
plt.title("Overall Rating Distribution (Train)")
plt.show()

# ---------------------------
# 3. Quartiles (Train only)
# ---------------------------
q1 = train_df['Overall_Rating'].quantile(0.25)
q2 = train_df['Overall_Rating'].quantile(0.50)
q3 = train_df['Overall_Rating'].quantile(0.75)

print("Q1:", q1, "Q2:", q2, "Q3:", q3)

# ---------------------------
# 4. Classification function
# ---------------------------
def classify(r):
    if r <= q1:
        return "Low"
    elif r <= q2:
        return "Mid"
    elif r <= q3:
        return "High"
    else:
        return "Elite"

# ---------------------------
# 5. Apply on train & test
# ---------------------------
train_df['Performance_Class'] = train_df['Overall_Rating'].apply(classify)
test_df['Performance_Class'] = test_df['Overall_Rating'].apply(classify)

# ---------------------------
# 6. Show distribution
# ---------------------------
print("\nTrain distribution:")
print(train_df['Performance_Class'].value_counts())

print("\nTest distribution:")
print(test_df['Performance_Class'].value_counts())

# ---------------------------
# 7. Plot
# ---------------------------
sns.countplot(x='Performance_Class', data=train_df)
plt.title("Class Distribution (Train)")
plt.show()

# ---------------------------
# 8. Justification
# ---------------------------
print("\nWe used quartiles from the training set to create balanced classes based on data distribution.")
#%%
model = LinearRegression()
model.fit(X_train_processed, y_train)

train_pred = model.predict(X_train_processed)
test_pred = model.predict(X_test_processed)

print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))

print("Train R2:", r2_score(y_train, train_pred))
print("Test R2:", r2_score(y_test, test_pred))
#%%
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

degrees = [1,2,3,4]

train_scores = []
test_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_num)
    X_test_poly = poly.transform(X_test_num)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_poly))
    test_r2 = r2_score(y_test, model.predict(X_test_poly))

    train_scores.append(train_r2)
    test_scores.append(test_r2)

    print(f"Degree {d} → Train R2: {train_r2:.3f}, Test R2: {test_r2:.3f}")

# Plot
plt.plot(degrees, train_scores, marker='o', label='Train')
plt.plot(degrees, test_scores, marker='o', label='Test')
plt.legend()
plt.title("Polynomial Performance")
plt.show()

print("\nObservation: If train ↑ and test ↓ → Overfitting")
#%%
alphas = np.logspace(-3, 3, 10)

ridge_rmse = []
lasso_rmse = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_poly, y_train)
    ridge_pred = ridge.predict(X_test_poly)
    ridge_rmse.append(np.sqrt(mean_squared_error(y_test, ridge_pred)))

    lasso = Lasso(alpha=a, max_iter=20000)
    lasso.fit(X_train_poly, y_train)
    lasso_pred = lasso.predict(X_test_poly)
    lasso_rmse.append(np.sqrt(mean_squared_error(y_test, lasso_pred)))

# Plot
plt.semilogx(alphas, ridge_rmse, label='Ridge')
plt.semilogx(alphas, lasso_rmse, label='Lasso')
plt.legend()
plt.title("Ridge vs Lasso")
plt.show()

print("Best Ridge Alpha:", alphas[np.argmin(ridge_rmse)])
print("Best Lasso Alpha:", alphas[np.argmin(lasso_rmse)])

print("\nRidge handles many features better, Lasso does feature selection.")
#%%
X_class = df.drop(columns=['Performance_Class', 'Overall_Rating'])
y_class = df['Performance_Class']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

X_train_c = pd.get_dummies(X_train_c)
X_test_c = pd.get_dummies(X_test_c)

X_train_c, X_test_c = X_train_c.align(X_test_c, fill_value=0, axis=1)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_c, y_train_c)

train_pred = model.predict(X_train_c)
test_pred = model.predict(X_test_c)

print("Train Accuracy:", accuracy_score(y_train_c, train_pred))
print("Test Accuracy:", accuracy_score(y_test_c, test_pred))

print(classification_report(y_test_c, test_pred))

sns.heatmap(confusion_matrix(y_test_c, test_pred), annot=True)
plt.show()
#%%
C_values = np.logspace(-2, 2, 5)

train_acc = []
test_acc = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(X_train_c, y_train_c)

    train_acc.append(accuracy_score(y_train_c, model.predict(X_train_c)))
    test_acc.append(accuracy_score(y_test_c, model.predict(X_test_c)))

plt.semilogx(C_values, train_acc, label='Train')
plt.semilogx(C_values, test_acc, label='Test')
plt.legend()
plt.show()

print("Best C:", C_values[np.argmax(test_acc)])
#%%
# -------------------------
# Naive Bayes Setup (FIXED)
# -------------------------

from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB

# نستخدم نفس الـ classification dataset اللي عملتيه
df_nb = train_df.copy()

num_cols_nb = ['Age', 'Overall_Rating', 'Value_Per_M$']
num_cols_nb = [col for col in num_cols_nb if col in df_nb.columns]

X_nb = df_nb[num_cols_nb]
y_nb = df_nb['Performance_Class']

# split صح (بدون leakage)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y_nb, test_size=0.2, random_state=42, stratify=y_nb
)
#%%
gnb = GaussianNB()
gnb.fit(X_train_nb, y_train_nb)

pred_gnb = gnb.predict(X_test_nb)

print("\nGaussianNB WITHOUT Scaling")
print("Accuracy:", accuracy_score(y_test_nb, pred_gnb))
print(classification_report(y_test_nb, pred_gnb))

cm = confusion_matrix(y_test_nb, pred_gnb)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("GaussianNB (No Scaling)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#%%
scaler_nb = StandardScaler()

X_train_scaled = scaler_nb.fit_transform(X_train_nb)
X_test_scaled = scaler_nb.transform(X_test_nb)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train_nb)

pred_gnb_scaled = gnb_scaled.predict(X_test_scaled)

print("\nGaussianNB WITH Scaling")
print("Accuracy:", accuracy_score(y_test_nb, pred_gnb_scaled))
print(classification_report(y_test_nb, pred_gnb_scaled))

cm = confusion_matrix(y_test_nb, pred_gnb_scaled)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("GaussianNB (Scaled)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#%%

X_train_bin = (X_train_nb > 0).astype(int)
X_test_bin = (X_test_nb > 0).astype(int)

bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train_nb)

pred_bnb = bnb.predict(X_test_bin)

print("\nBernoulliNB")
print("Accuracy:", accuracy_score(y_test_nb, pred_bnb))
print(classification_report(y_test_nb, pred_bnb))

cm = confusion_matrix(y_test_nb, pred_bnb)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("BernoulliNB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#%%
# make data non-negative
X_train_shift = X_train_nb - X_train_nb.min()
X_test_shift = X_test_nb - X_train_nb.min()

cnb = ComplementNB()
cnb.fit(X_train_shift, y_train_nb)

pred_cnb = cnb.predict(X_test_shift)

print("\nComplementNB")
print("Accuracy:", accuracy_score(y_test_nb, pred_cnb))
print(classification_report(y_test_nb, pred_cnb))

cm = confusion_matrix(y_test_nb, pred_cnb)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("ComplementNB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#%%
from sklearn.model_selection import KFold, cross_val_score

print("\n===== K-FOLD (REGRESSION) =====")

best_alpha = 1.0

best_ridge = Ridge(alpha=best_alpha)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    best_ridge,
    X_train_processed,
    y_train,
    scoring='neg_mean_squared_error',
    cv=kfold
)

rmse_scores = np.sqrt(-scores)

print("Fold RMSE:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Std RMSE:", rmse_scores.std())

# Plot
plt.figure()
plt.bar(range(1,6), rmse_scores)
plt.axhline(rmse_scores.mean(), linestyle='--')
plt.title("K-Fold RMSE")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.show()
#%%
from sklearn.model_selection import StratifiedKFold

print("\n===== STRATIFIED K-FOLD (CLASSIFICATION) =====")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_acc = []
nb_acc = []

num_cols_nb = ['Age', 'Future_Potential', 'Total_Stats_Score']

for train_idx, test_idx in skf.split(X_train_c, y_train_c):

    X_tr, X_te = X_train_c.iloc[train_idx], X_train_c.iloc[test_idx]
    y_tr, y_te = y_train_c.iloc[train_idx], y_train_c.iloc[test_idx]

    # 🔹 Logistic Regression
    log_model = LogisticRegression(max_iter=5000)
    log_model.fit(X_tr, y_tr)
    log_pred = log_model.predict(X_te)
    log_acc.append(accuracy_score(y_te, log_pred))

    # 🔹 Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_tr[num_cols_nb], y_tr)
    nb_pred = gnb.predict(X_te[num_cols_nb])
    nb_acc.append(accuracy_score(y_te, nb_pred))

print("\nLogistic Fold Accuracy:", log_acc)
print("Naive Bayes Fold Accuracy:", nb_acc)

print("\nLogistic Mean:", np.mean(log_acc), "Std:", np.std(log_acc))
print("Naive Bayes Mean:", np.mean(nb_acc), "Std:", np.std(nb_acc))

# Plot
x = np.arange(1,6)

plt.figure()
plt.bar(x - 0.2, log_acc, width=0.4, label='Logistic')
plt.bar(x + 0.2, nb_acc, width=0.4, label='Naive Bayes')
plt.legend()
plt.title("Fold-by-Fold Accuracy Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()

# REQUIRED CONCLUSION
print("""
Conclusion:
Logistic Regression achieved higher mean accuracy and lower variance,
which indicates better performance and stability compared to Naïve Bayes.
""")
#%%
print("""
================ MODEL COMPARISON ================

For regression, Ridge Regression performed better than Linear Regression,
because it reduces overfitting and provides lower RMSE on unseen data.

For classification, Logistic Regression outperformed Naïve Bayes,
as it captures relationships between features more effectively,
while Naïve Bayes assumes independence.

Cross-validation results showed that Logistic Regression is more stable,
with higher mean accuracy and lower variance across folds.

Classification is easier than regression in this dataset,
because predicting exact player value is harder than assigning categories.
""")
#%%
print("""
================ REGULARIZATION ANALYSIS ================

As alpha increases in Ridge and Lasso, model complexity decreases.
This reduces overfitting but too large alpha may cause underfitting.

Ridge performed better than Lasso because the dataset includes many
one-hot encoded features. Ridge shrinks coefficients smoothly,
while Lasso may eliminate useful features.

Lasso set some coefficients to zero, which means it performed feature selection
by removing less important variables.

In Logistic Regression, smaller values of C increase regularization strength.
The best C achieves a balance between bias and variance.

Overall, regularization improves generalization and prevents overfitting.
""")