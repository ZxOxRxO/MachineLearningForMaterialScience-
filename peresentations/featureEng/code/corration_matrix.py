import pandas as pd 
import numpy as np 


# get data xesl 
df = pd.read_excel ( 'drRezayat.xlsx')


correlation_mat = df.corr() 
df.corr().to_excel ( 'new_file.xlsx')
print ( ' file saved ?')