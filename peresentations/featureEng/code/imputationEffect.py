import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. ایجاد داده‌های اولیه با مقادیر پرت
np.random.seed(0)
data = np.random.randn(100, 1)

# ایجاد مقادیر پرت با ابعاد صحیح
outliers = np.random.randn(10, 1) * 10  # مقادیر پرت به ابعاد (10, 1)
data[::10] = outliers  # وارد کردن مقادیر پرت به داده‌ها

# تبدیل به DataFrame
df = pd.DataFrame(data, columns=["Feature"])

# 2. ایجاد مقادیر گم‌شده به صورت تصادفی
missing_rate = 0.2
missing_mask = np.random.rand(*df.shape) < missing_rate
df_missing = df.copy()
df_missing[missing_mask] = np.nan

# 3. استفاده از میانگین و میانه برای جاگذاری مقادیر گم‌شده
imputer_mean = SimpleImputer(strategy='mean')
df_mean_imputed = imputer_mean.fit_transform(df_missing)

imputer_median = SimpleImputer(strategy='median')
df_median_imputed = imputer_median.fit_transform(df_missing)

# 4. ایجاد داده‌های هدف
y = 2 * df.values.flatten() + 1 + np.random.randn(100)  # رگرسیون خطی با نویز

# تقسیم داده‌ها به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل با داده‌های اولیه
model.fit(X_train, y_train)
y_pred_initial = model.predict(X_test)

# آموزش مدل با داده‌های جاگذاری‌شده (میانه)
model.fit(X_train, y_train)
y_pred_median = model.predict(X_test)

# آموزش مدل با داده‌های جاگذاری‌شده (میانگین)
model.fit(X_train, y_train)
y_pred_mean = model.predict(X_test)

# 5. ترسیم نمودار اول (خط رگرسیون و داده‌های واقعی)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, label='Actual Data', color='black', alpha=0.6)
plt.plot(X_test, y_pred_initial, label='Regression Line (Initial)', color='blue', linewidth=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Regression Line with Actual Data')
plt.legend()

# 6. ترسیم نمودار دوم (مقایسه پیش‌بینی‌ها با روش‌های مختلف Imputation)
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, label='Actual Data', color='black', alpha=0.6)
plt.plot(X_test, y_pred_median, label='Prediction (Median Imputation)', linestyle='-.', color='green')
plt.plot(X_test, y_pred_mean, label='Prediction (Mean Imputation)', linestyle=':', color='red')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Predictions with Imputation Strategies')
plt.legend()

# نمایش نمودارها
plt.tight_layout()
plt.show()
