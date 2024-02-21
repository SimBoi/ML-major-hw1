import pyarrow as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import datetime
import time
from sklearn.model_selection import train_test_split


def convert_to_numerical_values(df):
    # blood type groups
    SpecialProperty = df['blood_type'].isin(['O+', 'B+'])
    SpecialProperty = [1 if specialProperty else -1 for specialProperty in SpecialProperty]
    df['SpecialProperty'] = SpecialProperty
    df.drop(['blood_type'], axis=1, inplace=True)

    # symptoms
    SymptomsSeverity = []
    for symptoms in df['symptoms']:
        if type(symptoms) == float:
            SymptomsSeverity.append(0)
        else:
            SymptomsSeverity.append(symptoms.count(';') + 1)
    df.drop(['symptoms'], axis=1, inplace=True)
    df['symptoms'] = SymptomsSeverity

    # sex
    NumericSex = df['sex'].isin(['M'])
    NumericSex = [1 if male == True else -1 for male in NumericSex]
    df.drop(['sex'], axis=1, inplace=True)
    df['sex'] = NumericSex

    # pcr date
    timestamps = []
    for date in df.pcr_date:
        timestamps.append(time.mktime(datetime.datetime.strptime(date, "%d-%m-%y").timetuple()) / 360)
    df.drop(['pcr_date'], axis=1, inplace=True)
    df['pcr_date'] = timestamps

    # current location
    x_location = []
    y_location = []
    for location in df.current_location:
        location = location.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(',')
        x_location.append(float(location[0]))
        y_location.append(float(location[1]))
    df.drop(['current_location'], axis=1, inplace=True)
    df['x_location'] = x_location
    df['y_location'] = y_location


def normalize(training_data, new_data):
    normalized_data = new_data.copy()

    # min max scaler
    scaler = sk.preprocessing.MinMaxScaler()
    columns = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_05', 'PCR_06', 'pcr_date']
    for column in columns:
        normalized_data.drop(column, inplace=True, axis=1)
        scaler.fit(training_data[column].values.reshape(-1, 1))
        normalized_data[column] = scaler.transform(new_data[column].values.reshape(-1, 1))

    # standard scaler
    scaler = sk.preprocessing.StandardScaler()
    columns = ['age', 'weight', 'num_of_siblings', 'happiness_score', 'household_income', 'conversations_per_day', 'sugar_levels', 'sport_activity', 'PCR_04', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10', 'symptoms', 'x_location', 'y_location']
    for column in columns:
        normalized_data.drop(column, inplace=True, axis=1)
        scaler.fit(training_data[column].values.reshape(-1, 1))
        normalized_data[column] = scaler.transform(new_data[column].values.reshape(-1, 1))

    return normalized_data


def prepare_data(training_data, new_data):
    training_data_copy = training_data.copy()
    new_data_copy = new_data.copy()
    convert_to_numerical_values(training_data_copy)
    convert_to_numerical_values(new_data_copy)
    return normalize(training_data_copy, new_data_copy)


def prepare_and_save_csv(input_csv, output_train_csv, output_test_csv, random_state=146):
    data = pd.read_csv(input_csv)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=random_state)
    train_df_prepared = prepare_data(train_df, train_df)
    test_df_prepared = prepare_data(train_df, test_df)
    train_df_prepared.to_csv(output_train_csv, index=False)
    test_df_prepared.to_csv(output_test_csv, index=False)
