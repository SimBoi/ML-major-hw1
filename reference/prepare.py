import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from google.colab import files
from scipy.spatial import distance
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

def normalize(training_data, new_data):

  new_dataset=new_data.copy()
  blood_type_groups=[]
  trainByBlood=new_dataset['blood_type'].values
  for i in range(len(new_dataset)):
    if trainByBlood[i]=='A+' or trainByBlood[i]=='A-':
      blood_type_groups.append(-1)
    elif trainByBlood[i]=='O+' or trainByBlood[i]=='O-':
      blood_type_groups.append(0)
    elif trainByBlood[i]=='B+' or trainByBlood[i]=='B-' or trainByBlood[i]=='AB+' or trainByBlood[i]=='AB-':
      blood_type_groups.append(1)

  new_dataset['blood_type_groups']=blood_type_groups
  new_dataset.drop("blood_type", axis=1,inplace=True)
  sore_throat=[]
  fever=[]
  smell_loss=[]
  cough=[]
  shortness_of_breath=[]
  arr= new_dataset['symptoms'].values.astype(str)

  for i in range(len(new_dataset)):
    if arr[i].__contains__('sore_throat'):
      sore_throat.append(1)
    else:
      sore_throat.append(0)
    if arr[i].__contains__('fever'):
      fever.append(1)
    else:
      fever.append(0)
    if arr[i].__contains__('smell_loss'):
      smell_loss.append(1)
    else:
      smell_loss.append(0)
    if arr[i].__contains__('cough'):
      cough.append(1)
    else:
      cough.append(0)
    if arr[i].__contains__('shortness_of_breath'):
      shortness_of_breath.append(1)
    else:
      shortness_of_breath.append(0)

  new_dataset['sore_throat']=sore_throat
  new_dataset['fever']=fever
  new_dataset['smell_loss']=smell_loss
  new_dataset['cough']=cough
  new_dataset['shortness_of_breath']=shortness_of_breath
  new_dataset.drop('symptoms',inplace=True,axis=1)
  new_dataset.loc[new_dataset["sex"] == "M", "sex"] = 1
  new_dataset.loc[new_dataset["sex"] == "F", "sex"] = -1


  pcr_days=[]
  newdatetrain=new_dataset['pcr_date'].values
  for i in range(len(new_dataset)):
      year, month , day = map(int, newdatetrain[i].split('-'))
      pcr_days.append(day + 30*month+(year-2019)*365)
      array_tmp=np.array(pcr_days)
  # array_tmp=array_tmp.reshape(-1,1)
  new_dataset=new_dataset.assign(normalized_pcr_date=preprocessing.MinMaxScaler().fit_transform(array_tmp.reshape(-1,1)))
  new_dataset.drop('pcr_date',inplace=True, axis=1)
  
  scaler = preprocessing.StandardScaler()
  newrray=np.copy(new_dataset['weight'])
  new_dataset.drop('weight',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(weight=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_01'])
  new_dataset.drop('PCR_01',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_01=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_02'])
  new_dataset.drop('PCR_02',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_02=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_03'])
  new_dataset.drop('PCR_03',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_03=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_04'])
  new_dataset.drop('PCR_04',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_04=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_05'])
  new_dataset.drop('PCR_05',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_05=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_06'])
  new_dataset.drop('PCR_06',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_06=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_07'])
  new_dataset.drop('PCR_07',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_07=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_08'])
  new_dataset.drop('PCR_08',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_08=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_09'])
  new_dataset.drop('PCR_09',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_09=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['PCR_10'])
  new_dataset.drop('PCR_10',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(PCR_10=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['household_income'])
  new_dataset.drop('household_income',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(household_income=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['sport_activity'])
  new_dataset.drop('sport_activity',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(sport_activity=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['sugar_levels'])
  new_dataset.drop('sugar_levels',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(sugar_levels=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['conversations_per_day'])
  new_dataset.drop('conversations_per_day',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(conversations_per_day=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['happiness_score'])
  new_dataset.drop('happiness_score',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(happiness_score=scaler.fit_transform(newrray))

  newrray=np.copy(new_dataset['age'])
  new_dataset.drop('age',inplace=True, axis=1)
  newrray=newrray.reshape(-1,1)
  new_dataset=new_dataset.assign(age=scaler.fit_transform(newrray))
  return new_dataset


def prepare_data(training_data, new_data):
  data = normalize(training_data, new_data)
  return data