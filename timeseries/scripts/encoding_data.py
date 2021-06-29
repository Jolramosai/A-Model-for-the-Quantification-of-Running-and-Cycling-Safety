import numpy as np 
import pandas as pd

file = '../datasets/timeseries_dataset.csv'

# Load data
def load_data():
    df = pd.read_csv(file)
    return df

'''
Function to encode all non-(int/float) features in a dataframe.
For each column, if its dtype is neither int or float, get the list of unique values,
store the relation between the label and the integer that encodes it and apply it.
Return a labelled dataframe and a dictionary label to be able to restore the original value. 
'''
def label_encoding(data):
    dic = {}
    for col in data.columns:
        if data[col].dtype == np.object:
            dic[col] = {}
    for col,diccol in dic.items():
        i = 0
        while i< data[col].value_counts().count():
            diccol[data[col].unique()[i]] = i
            i += 1
        
       
    df_labelled = data.replace(to_replace=dic, value=None)
    df_labelled.index = data['row ID']
    df_labelled.index.name = 'Index'
    
    return df_labelled, dic

def one_hot_encoding(data):
    columns = []
    for col in data.columns:
        if data[col].dtype == np.object:
            columns.append(col)
            
    data_pandas_ohe = pd.get_dummies(data,prefix=columns)
            
    return data_pandas_ohe
    

# Data contain all data readed
df = load_data()

# Encoding data(Label)
df_labelled, label_dictionary = label_encoding(df)
print(50*'-' + ' Valores codificados ' + '-'*50)
print('Unique count after Label Encoding: %d' %df_labelled['criteria_ozone'].value_counts().count())
print(df_labelled['criteria_ozone'].unique())

# Dictionary
print(50*'-' + ' Dicionário ' + '-'*50)
print(label_dictionary)

# Print dataset info
print(50*'-' + ' All data ' + '-'*50)
print(df_labelled.info())
print(110*'-')

# Encoding data(one_hot_encoding)
#df_labelled = one_hot_encoding(df)

# Save data labelled
df_labelled.drop(columns=['row ID'], inplace=True)
df_labelled.to_csv('../datasets/timeseries_final.csv', encoding='utf-8', index=True)
