#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
  data['year'] = data['date'].dt.dayofyear
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
  df_in = pd.read_csv(r'data/humid-temp-reorder_18-19.csv', encoding='iso-8859-1')
  df_out = pd.read_csv(r'data/Air_Q_18-19.csv')
  
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
  
  print(data_in)
  data_in = add_day_label(data_in)
  data_in = data_in.drop(data_in.columns[0], axis=1)
  data_in = data_in.drop(data_in.columns[5], axis=1)
  data_in = data_in.drop(data_in.index[0])
  #data_in = data_in.to_numpy()
  data_in = data_in.astype(float)
  print(data_in)

  del_column = [x for x in data_out.columns if x.find("pm25") == -1]
  #print(del_column)
  #print([data_out.columns])
  #print(type(data_out.columns))

  #print(data_out)
  data_out = data_out.drop(del_column, axis=1)
  data_out = data_out.drop(data_out.index[0])
  #data_out = data_out.to_numpy()
  data_out = data_out.astype(float)
  #print(data_out)

  test_in = pd.read_csv(r'data/test_in.csv')
  print(test_in)
  test_in = add_day_label(test_in)
  test_in = test_in.drop(test_in.columns[0], axis=1)
  test_in = test_in.drop(test_in.columns[5], axis=1)
  test_in = test_in.drop(test_in.index[0])
  #test_in = test_in.to_numpy()
  print(test_in)

  test_out = pd.read_csv(r'data/test_result.csv')
  test_out = test_out.drop(del_column, axis=1)
  test_out = test_out.drop(test_out.index[0])
  #test_out = test_out.to_numpy()

  # prepare data
  _max = []
  _min = []
  for x in data_in.columns:
    #print(x, ', ', data_in[x].max(), ', ', data_in[x].min())
    _max.append( data_in[x].max() if data_in[x].max() > test_in[x].max() else test_in[x].max() )
    _min.append( data_in[x].min() if data_in[x].min() < test_in[x].min() else test_in[x].min() )
  print(_max)
  print(_min)

  pm_max = []
  pm_min = []
  print("data_out = ", data_out.columns)
  indx = data_out.columns[0]
  pm_max.append( data_out[indx].max() if data_out[indx].max() > test_out[indx].max() else test_out[indx].max() )
  pm_min.append( data_out[indx].min() if data_out[indx].min() < test_out[indx].min() else test_out[indx].min() )
  #print(type(data_in[0]))
  #print(type(data_out[0]))
  #print(type(test_in[0]))
  #print(type(test_out[0]))
  print("========================")

  for i in range(data_in.shape[0]):
    for j in range(len(data_in.columns)):
      #ss = [type(data_in.iloc[i, j])]
      data_in.iloc[i, j] = ( data_in.iloc[i, j] - _min[j] ) / (_max[j] - _min[j])
      #ss.append(type(data_in.iloc[i, j]))
      #print(ss)

  for i in range(test_in.shape[0]):
    for j in range(len(test_in.columns)):
      test_in.iloc[i, j] = ( test_in.iloc[i, j] - _min[j] ) / (_max[j] - _min[j])

  '''
  for i in range(data_out.shape[0]):
    for j in range(len(data_out.columns)):
      data_out.iloc[i, j] = ( data_out.iloc[i, j] - _min[j] ) / (_max[j] - _min[j])


  for i in range(test_out.shape[0]):
    for j in range(len(test_out.columns)):
      test_out.iloc[i, j] = ( test_out.iloc[i, j] - _min[j] ) / (_max[j] - _min[j])

  #print(data_in)
  #print(data_out)
  #print(test_in)
  #print(test_out)
  #print(type(data_in[0]))
  #print(type(data_out[0]))
  #print(type(test_in[0]))
  #print(type(test_out[0]))
  '''
  #print(range(data_out.shape()))
  #print(range(data_out.shape[0]))

  print(data_out.shape)
  print(type(data_out))
  print(type(data_out.iloc[0, 0]))
  data_out = data_out.values.ravel()

  print(pm_max, ' : ', pm_min)
  for i in range(len(data_out)):
    ss = [type(data_out[i])]
    data_out[i] = ( data_out[i] - pm_min[0] ) / (pm_max[0] - pm_min[0])
    ss.append(type(data_out[i]))
    print(ss)

  #print(data_out)
  #data_out.iloc[0, 0] = data_out.iloc[0, 0]
  print(data_out.shape)
  print(type(data_out))
  print(type(data_out[0]))

  cnn = MLPClassifier()
  #cnn.fit(data_in, Y)
  #data_out = data_out.astype(np.float64)
  cnn.fit(data_in, data_out)
  #cnn.fit(scaler_in, scaler_out)
  #print(cnn)
  exit()

  Y_pred = []
  test_in = test_in.to_numpy()
  for i in range(len(test_in)):
    y_pred=(cnn.predict(test_in[[i]]))
    #print(y_pred)
    Y_pred.append(y_pred[0])
  print('predict = ', Y_pred)

  y = []
  err = []
  test_out = test_out.to_numpy()
  for i in range(len(test_out)):
    err.append(abs(test_out[i][0]-Y_pred[i]))
    y.append(test_out[i][0])
  print('actual = ', y)
  print('err = ', err, ' : ', sum(err))
  

  xx = range(len(err))
  fig = plt.figure('Prediction')
  plt.plot(xx, Y_pred, xx, y, xx, err)
  plt.legend(['predict', 'actual', 'error'])

  
  #fig = plt.figure('Data analysis')
  fig1, axarr = plt.subplots(3, 1)
  print(data_in)
  data_in = data_in.drop(data_in.columns[6], axis=1)
  data_in = data_in.drop(data_in.columns[4], axis=1)
  data_in = data_in.drop(data_in.columns[4], axis=1)
  x2 = range(len(data_in))
  print(data_in)
  humid = data_in
  humid = humid.drop(humid.columns[3], axis=1)
  humid = humid.drop(humid.columns[1], axis=1)
  humid = humid.drop(humid.columns[0], axis=1)
  axarr[0].plot(x2, humid, linewidth='0.5')
  axarr[0].legend(["Humidity (%)"])

  data_in = data_in.drop(data_in.columns[2], axis=1)
  axarr[1].plot(x2, data_in, linewidth='0.5')
  axarr[1].legend(["Temperature (° C)", "Dew Point (° C)", "Wind Speed (mph)"])

  axarr[2].plot(x2, data_out, linewidth='0.5')
  axarr[2].legend(["PM2.5"])
  
  reg = LinearRegression().fit(data_in, data_out.values.ravel())
  Y_pred = []
  for i in range(len(test_in)):
    y_pred=(cnn.predict(test_in[[i]]))
    #print(y_pred)
    Y_pred.append(y_pred[0])
  print('predict = ', Y_pred)
  y = []
  err = []
  for i in range(len(test_out)):
    err.append(abs(test_out[i][0]-Y_pred[i]))
    y.append(test_out[i][0])
  print('actual = ', y)
  print('err = ', err, ' : ', sum(err))
  
  
  plt.show()
