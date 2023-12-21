from flask import Flask, request, jsonify
import pandas as pd
from minio import Minio
import json

from loci import time_series as ts
import math
import panel as pn
pn.extension()

import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import altair as alt
import ipywidgets as widgets
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime

from scipy.spatial import distance
import plotly.graph_objects as go

from k_means_constrained import KMeansConstrained
from scipy.signal import savgol_filter

def train_meta_model(data, no_clusters, smooth, window, order, normalize, algorithms, params, gap_type, gap_length, gap_count):
    """
    Run the experiment for the imputation algorithms.
    :param data: dataframe with the data
    :param no_clusters: number of clusters to be used
    :param smooth: boolean to smooth the data
    :param window: window to be used for smoothing
    :param order: order to be used for smoothing
    :param normalize: boolean to normalize the data
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :param gap_type: type of gap to be generated
    :param gap_length: maximum length of the gap
    :param gap_count: maximum number of gaps
    :return: dataframe with the results
    """

    x = pd.DataFrame()
    
    # For each cluster
    for cl in range(no_clusters):
        
        ## Get a specific cluster ##
        print('\nRunning for cluster: ' + str(cl))
        try:
            data = data[data['cluster'] == cl]
            if len(data) == 0:
                print('Skipping empty cluster: ' + str(cl))
                continue
        except:
            continue
        
        data.drop('cluster', axis=1, inplace=True)
        data = data.T
        data.index = pd.to_datetime(data.index)

        # Normalize the data
        if normalize:
            data = z_normalize(data)

        # Smooth the data
        if smooth:
            data = smooth_data(data, window, order)

        # Generate custom gaps
        data_gaps = generate_custom_gaps(data, gap_length, gap_count, gap_type)
        training = data_gaps.copy()

        pred_ms_ts_original = run_imputation(training, algorithms, params)
        pred_ms_ts_original.replace([np.inf, -np.inf], np.nan, inplace=True)
        pred_ms_ts_original.dropna(inplace=True)
        pred_ms_ts_original = pred_ms_ts_original.loc[:, (pred_ms_ts_original != 0).any(axis=0)]
        pred_ms_ts_original = pred_ms_ts_original.loc[:, (pred_ms_ts_original != -1).any(axis=0)]
        column_names = pred_ms_ts_original.columns
        
        ## Train Ensemble Model ##
        real_np = data.to_numpy().flatten()
        new_np = training.to_numpy().flatten()
        real_val = real_np[np.isnan(new_np)]
        y = pd.Series(real_val)
        xgbr_model = XGBRegressor(learning_rate=0.1, max_depth=2, n_estimators=200, random_state=1, subsample=0.75)
        xgbr_model.fit(pred_ms_ts_original, y)

        # Only keep the gaps that are in the same cluster
        testing = data_gaps[data.columns]

        ## Apply imputation ##
        pred_ms_ts = run_imputation(testing, algorithms, params)
        pred_ms_ts = pred_ms_ts[column_names]
        
        # Estimate and Fill-in Missing Values
        y_pred = xgbr_model.predict(pred_ms_ts)
        x_curr = fill_missing_with_lm(testing, y_pred)

        ## Append to the Final Dataframes ##
        x = pd.concat([x, x_curr], axis=1)

    # Sort the data frame by column names and calculate metrics
    metrics = {}
    mae = np.mean(np.abs(x.values.flatten() - data.values.flatten()))
    rmse = np.sqrt(np.mean(np.square(x.values.flatten() - data.values.flatten())))
    metrics['mae'] = mae
    metrics['rmse'] = rmse

    return xgbr_model, metrics


def generate_custom_gaps(data, max_gap_length, max_gap_count, type):
    """
    Generate custom gaps for the data.
    :param data: dataframe with the data
    :param max_gap_length: maximum length of the gap
    :param max_gap_count: maximum number of gaps
    :param type: type of gap to be generated
    :return: dataframe with the data with gaps
    """

    data_gaps = data.copy()
    if type == 'random':
        for col in data_gaps.columns:
            num_missing = random.randint(1, max_gap_count)
            for i in range(num_missing):
                start = random.randint(0, len(data_gaps)-max_gap_length)
                end = start + random.randint(1, max_gap_length)
                if start == 0:
                    start = 1
                if end == len(data_gaps):
                    end = len(data_gaps)-1
                data_gaps[col][start:end] = np.nan
    if type == 'single':
        # TODO
        pass
    if type == 'no_overlap':
        # TODO
        pass
    if type == 'blackout':
        # TODO
        pass
    return data_gaps


def z_normalize(data):
    """
    Normalize the data.
    :param data: dataframe with the data
    :return: dataframe with the data normalized
    """
    return (data - data.mean()) / data.std()

def smooth_data(data, window, order):
    """
    Smooth the data.
    :param data: dataframe with the data
    :return: dataframe with the data smoothed
    """
    return data.apply(lambda x: savgol_filter(x, window, order))


def run_clustering(data, no_clusters):
    """
    Run clustering for the data.
    :param df: dataframe with the data
    :param no_clusters: number of clusters to be used
    :return: dataframe with the data and the clusters
    """

    # Perform k-means clustering the rows
    data_transposed = data.T
    kmeans = KMeansConstrained(n_clusters=no_clusters, random_state=0, size_min=3).fit(data_transposed)

    # Add cluster labels to dataframe
    data_transposed['cluster'] = kmeans.labels_
    data_final = data_transposed.copy()
    
    return data_final


def process_data(data):
    """
    Process the data to be used in the imputation algorithms.
    :param data: dataframe with the data
    :return: dataframe with the data processed
    """
    data_processed = data.copy()
    data_processed = data_processed.loc[:,~data_processed.columns.duplicated()] 
    data_processed['row_col'] = '[' + data_processed['row'].astype(str) + ',' + data_processed['col'].astype(str) + ']'
    data_processed.set_index('row_col', inplace=True)
    data_processed.drop(['row', 'col'], axis=1, inplace=True)
    data_processed.columns = pd.to_datetime(data_processed.columns, format='%Y%m%d') 
    data_processed = data_processed / 1000
    data_processed = data_processed.T
    data_processed.columns.name = None
    data_processed.index.names = ['date']
    data_processed.index = pd.to_datetime(data_processed.index, format='%y%m%d')
    data_processed = data_processed.loc[data_processed.index <= '2022-06-14']
    data_processed = data_processed.interpolate(method='linear', axis=1)
    data_processed.dropna(inplace=True, axis=1)

    return data_processed


def run_imputation(new_df, algorithms, params):
    """
    Run imputation for each algorithm and return a dataframe with the results.
    :param new_df: dataframe with the missing values
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :return: dataframe with the results
    """
    pred_ms_ts = pd.DataFrame()
    new_np = new_df.to_numpy().flatten()

    for alg in algorithms:
        new_alg_df = ts.fill_missing_values(new_df, alg, params)
        ts_np = new_alg_df.to_numpy().flatten()

        pred_ms_ts[str(alg)] = pd.Series(ts_np[np.isnan(new_np)])

    return pred_ms_ts

def fill_missing_with_lm(df: pd.DataFrame, lm_pred):
    """
    Fill missing values with the linear regression prediction.
    :param df: dataframe with the missing values
    :param lm_pred: linear regression prediction
    :return: dataframe with the missing values filled
    """
    new_df = df.copy()
    def is_nan(value):
        return math.isnan(float(value))

    i=0
    for index, row in new_df.iterrows():
        for j in range(len(row)):
            if is_nan(row.iloc[j]):
                row.iloc[j] = lm_pred[i]
                i = i + 1
        new_df.loc[index] = row
    return new_df