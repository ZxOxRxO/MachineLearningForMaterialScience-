import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. تولید داده‌های مصنوعی (سن و وزن)
np.random.seed(42)
n_samples = 500
age = np.random.randint(18, 80, size=n_samples)  # سن بین 18 تا 80 سال
weight = 0.5 * age + 20 + np.random.randn(n_samples) * 10  # رابطه خطی + نویز

# 2. افزودن مقادیر پرت (Outliers)
outlier_indices = np.random.choice(range(n_samples), size=20, replace=False)  # 20 داده پرت
weight[outlier_indices] = weight[outlier_indices] * 10  # ضرب وزن در 10 برای پرت

# 3. افزودن مقادیر گمشده (Missing Values)
missing_rate = 0.2  # 20 درصد مقادیر گم‌شده
missing_mask = np.random.rand(n_samples) < missing_rate
weight_with_missing = weight.copy()
weight_with_missing[missing_mask] = np.nan

# تبدیل به DataFrame
df = pd.DataFrame({'Age': age, 'Weight': weight_with_missing})

# 4. روش‌های Imputation
# - با میانگین (Mean)
imputer_mean = SimpleImputer(strategy='mean')
df_mean = df.copy()
df_mean['Weight'] = imputer_mean.fit_transform(df[['Weight']])

# - با میانه (Median)
imputer_median = SimpleImputer(strategy='median')
df_median = df.copy()
df_median['Weight'] = imputer_median.fit_transform(df[['Weight']])

# 5. رگرسیون خطی برای هر دو روش
def train_and_predict(df, label):
    X = df['Age'].values.reshape(-1, 1)
    y = df['Weight'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return X, y, y_pred, model

# - با میانگین
X_mean, y_mean, y_pred_mean, model_mean = train_and_predict(df_mean, 'Mean Imputation')

# - با میانه
X_median, y_median, y_pred_median, model_median = train_and_predict(df_median, 'Median Imputation')

# 6. نمایش نتایج در نمودار
plt.figure(figsize=(15, 6))

# نمودار داده‌های میانگین
plt.subplot(1, 2, 1)
plt.scatter(X_mean, y_mean, label='Data (Mean Imputation)', color='blue', alpha=0.5)
plt.plot(X_mean, y_pred_mean, label='Regression Line (Mean)', color='red', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Prediction with Mean Imputation')
plt.legend()

# نمودار داده‌های میانه
plt.subplot(1, 2, 2)
plt.scatter(X_median, y_median, label='Data (Median Imputation)', color='green', alpha=0.5)
plt.plot(X_median, y_pred_median, label='Regression Line (Median)', color='orange', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Prediction with Median Imputation')
plt.legend()

plt.tight_layout()
plt.show()
