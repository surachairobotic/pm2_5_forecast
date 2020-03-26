#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier


def for_bangkok_air_quality():
  df = pd.read_csv(r'data/bangkok-air-quality.csv')
  print(df)
  print(df.shape)
  for i in range(df.shape[0]):
    txt = df.iloc[i, 0].split("/")
    for j in range(len(txt)):
      if len(txt[j]) < 2:
        txt[j] = '0'+txt[j]
    df.iloc[i, 0] = txt[0]+'/'+txt[1]+'/'+txt[2]
  print(df)
  df.sort_values(by=['date'], inplace=True, ascending=True)
  df.replace(' ', ' -1', inplace=True)
  print(df)
  df.to_csv(r'data/bangkok-air-quality-reorder.csv', index = False, header=True)

def for_P_data(f_name):
  df = pd.read_csv(r'data/p18_11.csv', encoding='iso-8859-1')
  #print(df)
  f_name = '20'+f_name[1:6].replace('_', '/')
  #print(f_name)
  for i in range(df.shape[0]):
    if df.iloc[i, 0] < 10:
      df.iloc[i, 0] = '0'+str(df.iloc[i, 0])
    df.iloc[i, 0] = f_name+'/'+str(df.iloc[i, 0])
  return df

def get_files_list(mypath):
  from os import listdir
  from os.path import isfile, join
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  return onlyfiles

def add_day_label(data):
  data['date'] = pd.to_datetime(data['date'], errors='coerce')
  data['weekday'] = data['date'].dt.dayofweek
  return data

if __name__ == '__main__':
  '''
  for_bangkok_air_quality()
  
  files_list = get_files_list('data')
  files_list = [x for x in files_list if x[0] is 'p']
  print(files_list)
  l_df = []
  for f in files_list:
    l_df.append(for_P_data(f))
    
  df_new = pd.concat([x for x in l_df])
  print(df_new)
  df_new.rename(columns={'Time':'date'}, inplace=True)
  df_new.sort_values(by=['date'], inplace=True, ascending=True)
  df_new.replace(' ', ' -1', inplace=True)
  print(df_new)
  df_new.to_csv(r'data/humid-temp-reorder.csv', index = False, header=True)
  '''  
  '''
  df_in = pd.read_csv(r'data/humid-temp-reorder.csv')
  df_out = pd.read_csv(r'data/bangkok-air-quality-reorder.csv')
  
  duplicates_in = set(df_in.index).intersection(df_out.index)
  print(duplicates_in)
  duplicates_out = set(df_out.index).intersection(df_in.index)
  
  df_in = df_in.drop(duplicates_in, axis=0)
  df_out = df_out.drop(duplicates_out, axis=0)
  
  df_in.to_csv(r'data/input.csv', index = False, header=True)
  df_out.to_csv(r'data/output.csv', index = False, header=True)
  '''
    
  data_in = pd.read_csv(r'data/input.csv')
  data_out = pd.read_csv(r'data/output.csv')
  
  data_in = add_day_label(data_in)
  data_in = data_in.drop(data_in.columns[0], axis=1)
  data_in = data_in.drop(data_in.index[0])
  print(data_in)

  del_column = [x for x in data_out.columns if x.find("pm25") == -1]
  print(del_column)
  print([data_out.columns])
  print(type(data_out.columns))

  data_out = data_out.drop(del_column, axis=1)
  data_out = data_out.drop(data_out.index[0])

  print(data_in)
  print(data_out)

  cnn = MLPClassifier()
  cnn.fit(data_in, data_out.values.ravel())
  print(cnn)
  
  test_in = pd.read_csv(r'data/test_in.csv')
  test_in = add_day_label(test_in)
  test_in = test_in.drop(test_in.columns[0], axis=1)
  test_in = test_in.drop(test_in.index[0])
  test_in = test_in.to_numpy()
  print(test_in)

  Y_pred = []
  for i in range(len(test_in)):
    y_pred=(cnn.predict(test_in[[i]]))
    #print(y_pred)
    Y_pred.append(y_pred[0])
  print(Y_pred)

  test_out = pd.read_csv(r'data/test_result.csv')
  test_out = test_out.drop(del_column, axis=1)
  test_out = test_out.drop(test_out.index[0])
  test_out = test_out.to_numpy()
  y = []
  for x in test_out:
    y.append(x[0])
  print(y)
