# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

# =============================================================================
# 2. LOAD DATA
# =============================================================================
# Example: marketing sales data
# d_sales = pd.read_csv('marketing_sales_data.csv')

# Example: diamonds dataset for demonstration
d = sns.load_dataset("diamonds", cache=False)

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Peek at the first rows and category counts
print(d.head())
print(d['color'].value_counts())
# Check for missing values
print(d.isna().sum())
# Visualize pairwise relationships
sns.pairplot(d)
plt.show()

# =============================================================================
# 4. DATA CLEANING & FEATURE ENGINEERING
# =============================================================================
# Filter to a subset of colors and keep only price
colorless = (
    d[d["color"].isin(["E", "F", "H", "D", "I"])]
    [["color", "price"]]
    .reset_index(drop=True)
)

# Convert 'color' to a categorical and drop unused levels
colorless['color'] = colorless['color'].astype('category')
colorless['color'] = colorless['color'].cat.remove_unused_categories()

# Add a log-transformed price column
colorless['log_price'] = np.log(colorless['price'])

# Drop any remaining NaNs and reset index
colorless.dropna(inplace=True)
colorless.reset_index(drop=True, inplace=True)

# Save cleaned dataset
colorless.to_csv('diamonds_clean.csv', index=False)

# =============================================================================
# 5. ONE-WAY ANOVA (Sales ~ TV)
# =============================================================================
# (Uncomment and load your actual sales data into d_sales first)
# model_sales = ols('Sales ~ C(TV)', data=d_sales).fit()
# print("Type I ANOVA:\n", sm.stats.anova_lm(model_sales, typ=1))
# print("Type II ANOVA:\n", sm.stats.anova_lm(model_sales, typ=2))
# print("Type III ANOVA:\n", sm.stats.anova_lm(model_sales, typ=3))

# =============================================================================
# 6. RESIDUAL DIAGNOSTICS
# =============================================================================
# resid = model_sales.resid
# # Histogram + Q-Q plot
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# sns.histplot(resid, ax=ax[0], kde=True).set_title('Residuals Histogram')
# sm.qqplot(resid, line='s', ax=ax[1]).set_title('Normal Q-Q')
# plt.tight_layout()
# plt.show()
# # Fitted vs. Residuals
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x=model_sales.fittedvalues, y=resid)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel('Fitted Values'); plt.ylabel('Residuals')
# plt.title('Residuals vs. Fitted')
# plt.show()

# =============================================================================
# 7. POST HOC: TUKEY HSD
# =============================================================================
# tukey_results = pairwise_tukeyhsd(
#     endog=d_sales['Sales'],
#     groups=d_sales['TV'],
#     alpha=0.05
# )
# print(tukey_results.summary())

# =============================================================================
# 8. LOGISTIC REGRESSION TEMPLATE
# =============================================================================
# -- Prepare df_s with 'Inflight entertainment' and binary 'satisfaction_enc' --
# df_s = pd.read_csv('your_data.csv')
# encoder = OneHotEncoder(drop='first', sparse=False)
# df_s['satisfaction_enc'] = encoder.fit_transform(df_s[['satisfaction']])[:, 0]

# X = df_s[['Inflight entertainment']]
# y = df_s['satisfaction_enc']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# clf = LogisticRegression().fit(X_train, y_train)

resample(data_minority,
                                 replace=True,                 # to sample with replacement
                                 n_samples=len(data_majority), # to match majority class
                                 random_state=0)


# # Predictions & probabilities
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)[:, 1]

# # Evaluation metrics
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Recall:", metrics.recall_score(y_test, y_pred))
# print("F1 Score:", metrics.f1_score(y_test, y_pred))

# # Plot logistic regression fit
# sns.regplot(
#     x="Inflight entertainment",
#     y="satisfaction_enc",
#     data=df_s,
#     logistic=True,
#     ci=None
# )
# plt.title('Logistic Regression Fit')
# plt.show()
