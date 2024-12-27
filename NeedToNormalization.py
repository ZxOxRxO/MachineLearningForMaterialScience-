import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score




np.random.seed(42)
X = np.random.rand(100, 1) * 10  # ویژگی (Feature)
y = 2 * X.flatten() + np.random.normal(0, 5, 100) ** 2  # متغیر هدف چوله (Skewed Target)
data = pd.DataFrame({'Feature': X.flatten(), 'Target': y})



# data set spliting :) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# First model withou Target Transformation ( without Normalization )
model_1 = LinearRegression()
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

# secend model withou Target Transformation ( with  Normalization )
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
model_2 = LinearRegression()
model_2.fit(X_train, y_train_log)
y_pred_2 = model_2.predict(X_test)
y_pred_2_exp = np.expm1(y_pred_2)  

# mse calculation for each model :) 
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)
mse_2 = mean_squared_error(y_test, y_pred_2_exp)
r2_2 = r2_score(y_test, y_pred_2_exp)

# visualization 
plt.figure(figsize=(16, 6))

# first model ( without normalization ) visualization 
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test.flatten(), y=y_test, color='blue', label='Actual')
sns.lineplot(x=X_test.flatten(), y=y_pred_1, color='red', label='Predicted')
plt.title(f'Without Transformation\nMSE: {mse_1:.2f}, R2: {r2_1:.2f}')
plt.xlabel('Feature')
plt.ylabel('Target')

# secend model visualization :) 
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test.flatten(), y=y_test, color='blue', label='Actual')
sns.lineplot(x=X_test.flatten(), y=y_pred_2_exp, color='green', label='Predicted')
plt.title(f'With Log Transformation\nMSE: {mse_2:.2f}, R2: {r2_2:.2f}')
plt.xlabel('Feature')
plt.ylabel('Target')

plt.tight_layout()
plt.show()
