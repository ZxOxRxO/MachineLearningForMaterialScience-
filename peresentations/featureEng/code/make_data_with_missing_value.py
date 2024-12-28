import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 500

data = {
    'Home_Area_sqft': np.random.randint(500, 5000, size=n_rows),
    'Num_Bedrooms': np.random.randint(1, 6, size=n_rows),
    'Num_Bathrooms': np.random.randint(1, 4, size=n_rows),
    'Location_Score': np.random.uniform(1, 10, size=n_rows),
    'Home_Price': np.random.randint(50000, 500000, size=n_rows)
}

df = pd.DataFrame(data)

missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices, 'Location_Score'] = np.nan

df.to_excel('home_price_with_missing_valu.xlsx', index=False)

print("Excel file 'home_price_data.xlsx' with synthetic home price data and missing values created successfully!")
