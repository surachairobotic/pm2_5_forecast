#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
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

