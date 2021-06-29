import numpy as np 
import pandas as pd

file = '../datasets/final/dataset_final.csv'

# Load data
def load_data():
    df = pd.read_csv(file)
    return df

# Data contain all data readed
df = load_data()

# Print dataset info
print(50*'-' + ' All data ' + '-'*50)
print(df.info())
print(110*'-')

# Infer objects type
df.infer_objects()

# Get missings columns
print(47*'-' + ' Missing columns ' + '-'*47)
missings = df.columns[df.isnull().any()].tolist()
print(missings)
print(110*'-')

# Check and replace missing with -99 (masking)
print(45*'-' + ' Num missing values ' + '-'*45)
print(df.isnull().sum()) 
df.fillna(-99, inplace=True)
print(110*'-')

# Frequency distribution of columns (missings)
print(47*'-' + ' Missing values ' + '-'*47)
for category in missings:
    print('-> Column:', category)
    print(df[category].unique())
    print('Unique count: %d' %df[category].value_counts().count()) 
    print(df[category].value_counts())
    print(50*'-')
print(110*'-')
