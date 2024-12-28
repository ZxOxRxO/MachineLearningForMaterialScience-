import pandas as pd

# Load the Excel file
file_path = 'home_price_with_missing_valu.xlsx'  # Replace with your file name
df = pd.read_excel(file_path)


for column in df.select_dtypes(include='number').columns: 

    median = 0 
     
    df[column].fillna(median, inplace=True)


output_file_path = 'newfile.xlsx'  
df.to_excel(output_file_path, index=False)

print(f"File saved as {output_file_path}")
