import copy
import json
import math
import random
import string
from datetime import datetime
from os import walk
import algorithms as alg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import ruptures as rpt
import seaborn as sns
from PIL import ImageColor
#from fbprophet import Prophet
from scipy.stats import anderson
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess
#from tsfresh.feature_extraction import extract_features
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statistics import mean


class DictionarySizeIsNotSupported(Exception): pass


class StringsAreDifferentLength(Exception): pass


class OverlapSpecifiedIsNotSmallerThanWindowSize(Exception): pass


#pyo.init_notebook_mode()


# ---------------------------------------------------------#
# ------------------------- SAX ---------------------------#
# --- source: https://github.com/serval-snt-uni-lu/dsco ---#
# ---------------------------------------------------------#
class __SAX(object):
    """
    This class calculates the Symbolic Aggregate approXimation method (SAX) word of a given time series. In short, this translates a series of data to a string, which can then be compared with other such strings using a lookup table.
    
    Source: https://github.com/serval-snt-uni-lu/dsco
    
    :param wordSize: The SAX word length.
    :type wordSize: int
    :param alphabetSize: The alphabet size for SAX words.
    :type alphabetSize: int
    :param epsilon: Used for normalization.
    :type epsilon: float
    """

    def __init__(self, wordSize=8, alphabetSize=7, epsilon=1e-6):

        if alphabetSize < 3 or alphabetSize > 20:
            raise DictionarySizeIsNotSupported()
        self.aOffset = ord('a')
        self.wordSize = wordSize
        self.alphabetSize = alphabetSize
        self.eps = epsilon
        self.breakpoints = {'3': [-0.43, 0.43],
                            '4': [-0.67, 0, 0.67],
                            '5': [-0.84, -0.25, 0.25, 0.84],
                            '6': [-0.97, -0.43, 0, 0.43, 0.97],
                            '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                            '9': [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                            '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                            '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                            '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                            '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                            '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                            '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11,
                                   1.5],
                            '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89,
                                   1.15, 1.53],
                            '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72,
                                   0.93, 1.19, 1.56],
                            '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59,
                                   0.76, 0.97, 1.22, 1.59],
                            '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48,
                                   0.63, 0.8, 1, 1.25, 1.62],
                            '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39,
                                   0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
                            }
        self.beta = self.breakpoints[str(self.alphabetSize)]
        self.build_letter_compare_dict()
        self.scalingFactor = 1

    def get_breakpoints(self):
        """
        Returns the SAX breakpoint set.
        
        :return: A dictionary containing the breakpoints.
        :rtype: dict
        """
        return self.beta.copy()

    def to_letter_rep(self, x):
        """
        This function takes a series of data, x, and transforms it to a string representation.
        
        :param x: The input data.
        :type x: list
        :return: The corresponding string representation.
        :rtype: string
        """
        (paaX, indices) = self.to_PAA(self.normalize(x))
        self.scalingFactor = np.sqrt((len(x) * 1.0) / (self.wordSize * 1.0))
        return (self.alphabetize(paaX), indices)

    def normalize(self, x):
        """
        This function will z-normalize an array (give it a mean of 0, and a standard deviation of 1) unless it's standard deviation is below epsilon, in which case it returns an array of zeros the length of the original array.
        
        :param x: The input array.
        :type x: list
        :return: The z-normalized array.
        :rtype: list
        """
        X = np.asanyarray(x)
        if np.nanstd(X) < self.eps:
            res = []
            for entry in X:
                if not np.isnan(entry):
                    res.append(0)
                else:
                    res.append(np.nan)
            return res
        return (X - np.nanmean(X)) / np.nanstd(X)

    def to_PAA(self, x):
        """
        This function performs Piecewise Aggregate Approximation (PAA) on data set, reducing the dimension of the dataset x to w discrete levels. It returns the reduced dimension data set, as well as the indices corresponding to the original data for each reduced dimension
        
        :param x: The input array.
        :type x: list
        :return: The PAA array.
        :rtype: list
        """
        n = len(x)
        stepFloat = n / float(self.wordSize)
        step = int(math.ceil(stepFloat))
        frameStart = 0
        approximation = []
        indices = []
        i = 0
        while frameStart <= n - step:
            thisFrame = np.array(x[frameStart:int(frameStart + step)])
            approximation.append(np.mean(thisFrame))
            indices.append((frameStart, int(frameStart + step)))
            i += 1
            frameStart = int(i * stepFloat)
        return (np.array(approximation), indices)

    def alphabetize(self, paaX):
        """
        This function converts the Piecewise Aggregate Approximation (PAA) of x to a series of letters.
        
        :param paaX: The input PAA array.
        :type paaX: list
        :return: The resulting series of letters.
        :rtype: list
        """
        alphabetizedX = ''
        for i in range(0, len(paaX)):
            letterFound = False
            for j in range(0, len(self.beta)):
                if np.isnan(paaX[i]):
                    alphabetizedX += '-'
                    letterFound = True
                    break
                if paaX[i] < self.beta[j]:
                    alphabetizedX += chr(self.aOffset + j)
                    letterFound = True
                    break
            if not letterFound:
                alphabetizedX += chr(self.aOffset + len(self.beta))
        return alphabetizedX

    def compare_strings(self, sA, sB):
        """
        This function compares two strings based on individual letter distance. It requires that both strings are the same length.
        
        :param sA: The first string.
        :type sA: string
        :param sB: The second string.
        :type sB: string
        :return: The resulting letter distance.
        :rtype: float
        """
        if len(sA) != len(sB):
            raise StringsAreDifferentLength()
        list_letters_a = [x for x in sA]
        list_letters_b = [x for x in sB]
        mindist = 0.0
        for i in range(0, len(list_letters_a)):
            if list_letters_a[i] != '-' and list_letters_b[i] != '-':
                mindist += self.compare_letters(list_letters_a[i], list_letters_b[i]) ** 2
        mindist = self.scalingFactor * np.sqrt(mindist)
        return mindist

    def compare_letters(self, la, lb):
        """
        This function compares two letters based on their letter distance.
        
        :param la: The first letter.
        :type la: string
        :param lb: The second letter.
        :type lb: string
        :return: The resulting letter distance.
        :rtype: float
        """
        return self.compareDict[la + lb]

    def build_letter_compare_dict(self):
        """
        This function builds up the lookup table to determine numeric distance between two letters given an alphabet size.  Entries for both 'ab' and 'ba' will be created and will have identical values.
        
        :return: None.
        :rtype: none
        """

        number_rep = range(0, self.alphabetSize)
        letters = [chr(x + self.aOffset) for x in number_rep]
        self.compareDict = {}
        for i in range(0, len(letters)):
            for j in range(0, len(letters)):
                if np.abs(number_rep[i] - number_rep[j]) <= 1:
                    self.compareDict[letters[i] + letters[j]] = 0
                else:
                    high_num = np.max([number_rep[i], number_rep[j]]) - 1
                    low_num = np.min([number_rep[i], number_rep[j]])
                    self.compareDict[letters[i] + letters[j]] = self.beta[high_num] - self.beta[low_num]


# -------------------------------------#
# ------ OTHER TS FUNCTIONS -----------#
# -------------------------------------#
def read_files(my_path, date_column, data_column, date_format='%m/%d/%Y'):
    """
    Reads all the co-evolving time series files contained in the given path in a Pandas dataframe.
    
    :param my_path: The path containing the time series files.
    :type my_path: string
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :param date_format: The format of the date.
    :type date_format: string
    :return:
        -df_array (:py:class:'list') - A list containing a Pandas dataframe for each read time series.
        -filenames (:py:class:'list') - A list containing all read filenames.
        -start_date (:py:class:'datetime') - The starting datetime of all time series.
        -end-date (:py:class:'datetime') - The ending datetime of all time series.
    """
    df_array = []
    try:
        filenames = next(walk(my_path), (None, None, []))[2]
    except:
        return -1, -1, -1

    # print(my_path)
    start_date = None
    end_date = None
    if len(filenames) > 0:
        count = 0
        for filename in filenames:
            if filename == '.DS_Store':
                continue
            try:
                df = pd.read_csv(my_path + filename, header=None)
            except:
                continue

            if count == 0:
                start_date = df.iloc[0][date_column]
                end_date = df.iloc[-1][date_column]
                count += 1

            df['normalized'] = zscore(df[data_column])
            df_array.append(df)

    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)
    return df_array, filenames, start_date, end_date


def read_file(my_file):
    """
    Reads a single time series from the given file.
    
    :param my_file: The file
    :type my_file: string
    :return: A Pandas dataframe containing the read time series.
    :rtype: pandas.DataFrame
    """
    try:
        df = pd.read_csv(my_file, header=None)
    except:
        return pd.DataFrame()

    return df


def __get_date_range(start, end, intv, date_format='%m/%d/%Y'):
    """
    Returns a list of dates within a given range.
    
    :param start: The starting date.
    :type start: datetime
    :param end: The ending date.
    :type end: datetime
    :param intv: The interval between dates (in days).
    :type intv: int
    :param date_format: The format of the date.
    :type date_format: string
    :return: A list of dates.
    :rtype: list
    """
    #    start = datetime.strptime(start,"%m/%d/%Y")
    #    end = datetime.strptime(end,"%m/%d/%Y")
    diff = (end - start) / intv
    for i in range(intv):
        yield (start + diff * i).strftime(date_format)
    yield end.strftime(date_format)


def create_sankey(df_array, alphabet_size, word_size, begin, end, from_value=float('-inf'), to_value=float('+inf'),
                  date_format='%m/%d/%Y'):
    """
    This method generates and returns an interactive, Plotly-based SankeyTS diagram. The bands and flows of the diagram are generated based on the SAX words of the loaded set of time series.
    
    :param df_array: A list containing a Pandas dataframe for each read time series.
    :type df_array: list
    :param alphabet_size: The alphabet size for SAX encoding.
    :type alphabet_size: int
    :param word_size: The SAX word length.
    :type word_size: int
    :param begin: The starting date for the SankeyTS diagram.
    :type begin: string
    :param end: The ending date for the SankeyTS diagram.
    :type end: string
    :param from_value: The starting value for the SankeyTS diagram.
    :type from_value: float
    :param to_value: The ending value for the SankeyTS diagram.
    :type to_value: float
    :param date_format: The format of the date.
    :type date_format: string
    :return: A SankeyTS diagram.
    :rtype: plotly.graph_objects.figure
    """
    sax = __SAX(wordSize=word_size, alphabetSize=alphabet_size)
    breakpoints = sax.get_breakpoints()
    breakpoints.insert(0, float("-inf"))
    breakpoints.append(float("+inf"))
    begin = datetime.strptime(begin, date_format).date()
    end = datetime.strptime(end, date_format).date()
    date_range = list(__get_date_range(begin, end, word_size))
    alphabet = dict(enumerate(string.ascii_lowercase))
    alphabet_inv = dict(zip(string.ascii_lowercase, range(0, 26)))

    label = []
    node_x = []
    node_y = []
    step_x = 1 / (word_size)
    step_y = 1 / (alphabet_size)
    curr_x = 0
    curr_y = 0
    dates = []
    breakpoint_intervals = []
    for i in range(0, word_size):
        curr_x = i * step_x
        for j in range(0, alphabet_size):
            curr_y = j * step_y
            label.append(alphabet[j])
            node_x.append(curr_x + 0.01)
            node_y.append(curr_y + 0.01)
            dates.append(date_range[i])
            breakpoint_intervals.append(
                'from ' + str(breakpoints[alphabet_size - j - 1]) + ' to ' + str(breakpoints[alphabet_size - j]))

    source_target_value = {}
    node_set = []
    for df in df_array:
        df[0] = pd.to_datetime(df[0], format=date_format).dt.date
        mask = (df[0] >= begin) & (df[0] <= end)
        df = df.loc[mask]

        df = df[df['normalized'] >= from_value]
        df = df[df['normalized'] <= to_value]
        #         if df['normalized'].mean() < from_value or df['normalized'].mean() > to_value:
        #             continue
        df = df.drop(['normalized'], axis=1)
        signal = df.iloc[:, 2:].values
        sax_rep = sax.to_letter_rep(signal)[0]

        for i in range(1, len(sax_rep)):
            source_target = (
                alphabet_inv[sax_rep[i - 1]] + ((i - 1) * alphabet_size),
                alphabet_inv[sax_rep[i]] + (i * alphabet_size))
            if source_target not in source_target_value.keys():
                source_target_value[source_target] = 1
            else:
                source_target_value[source_target] = source_target_value[source_target] + 1
            node_set.append(source_target[0])
            node_set.append(source_target[1])

    node_x_pos = []
    node_y_pos = []
    for i in set(node_set):
        node_x_pos.append(node_x[i])
        node_y_pos.append(node_y[i])

    source = []
    target = []
    value = []
    for key in source_target_value:
        source.append(key[0])
        target.append(key[1])
        value.append(int(source_target_value[key]))

    pal1 = sns.color_palette("Spectral", alphabet_size)
    pal1 = pal1.as_hex()

    node_colors = []
    for i in range(0, len(label)):
        node_colors.append(pal1[int(i % alphabet_size)])

    link_colors = [node_colors[src] for src in source]
    link_breakpoint_intervals = [breakpoint_intervals[src] for src in source]

    link_colors_rgb = []
    count = 0
    for color in link_colors:
        rgb = ImageColor.getcolor(color, "RGB")
        link_colors_rgb.append('rgba(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ',' + str(
            value[count] / (max(value))) + ')')
        count += 1

    fig = go.Figure(data=[go.Sankey(
        valueformat=".0f",
        textfont=dict(color="rgba(0,0,0,0)", size=1),
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=1.75),
            label=dates,
            customdata=breakpoint_intervals,
            x=node_x_pos,
            y=node_y_pos,
            color=node_colors,
            hovertemplate='Value range %{customdata}<extra></extra>',
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            line=dict(color="black", width=0.75),
            color=link_colors_rgb,
            customdata=link_breakpoint_intervals,
            hovertemplate='Count: %{value} <br />From %{source.label} to %{target.label}<extra></extra>',
        ))])

    fig.update_layout(autosize=True, font_size=16)

    return fig


def __get_gain_index(trend, seasonality, model, real_values):
    """
    This method takes as input the trend, seasonality the model type as well as the real values of the time series and calculates the gain index. The gain index expresses the contribution of the seasonality to the actual time series. In order to calculate it we calculate the percentage difference between the percentage error of trend vs actual time series and percentage error of trend + seasonality vs actual time series. The higher this index is, the more likely is for the time series to have a strong seasonality component.
    
    :param trend: The Trend component.
    :type trend: list
    :param seasonality: The Seasonal component.
    :type seasonality: list
    :param model: The type of the model, 'additive' or 'multiplicative'.
    :type model: string
    :return: The gain index, a percentage which expresses the contribution of the seasonal component.
    :rtype: float
    """
    error1 = []  # error between trend vs actual values
    error2 = []  # error between trend + seasonality vs actual values

    for i in range(0, len(real_values)):
        if trend[i] != None and real_values[i] != 0:
            if model == 'Additive':
                predicted = trend[i]
                real = real_values[i]
                error1.append(abs(real - predicted) / abs(real))

                predicted = trend[i] + seasonality[i]
                real = real_values[i]
                error2.append(abs(real - predicted) / abs(real))

            if model == 'Multiplicative':
                predicted = trend[i]
                real = real_values[i]
                error1.append(abs(real - predicted) / abs(real))

                predicted = trend[i] * seasonality[i]
                real = real_values[i]
                error2.append(abs(real - predicted) / abs(real))

    error = (mean(error1) - mean(error2)) / mean(error1)

    return error


def __extract_best_period(ts, dates, periods, model):
    """
    This method decomposes the provided time series using all the provided periods and selects the one that gives the highest gain index. It also returns the calculated gain indexes for all the tested periods. The gain index expresses the contribution of the seasonality to the actual time series. In order to calculate it we calculate the percentage difference between the percentage error of trend vs actual time series and percentage error of trend + seasonality vs actual time series. The higher this index is the more likely is for the time series to have a strong seasonality component.
    
    :param ts:A list containing the time series values.
    :type ts: list
    :param dates: A list of datetime objects corresponding to the time series values.
    :type dates: list
    :param periods: A list containing the periods to be tested.
    :type periods: list
    :param model: The type of the model, 'additive' or 'multiplicative'.
    :type model: string
    :return:
        -p (:py:class:'int') - The selected period (the one that gives the highest gain index).
        -periods_gain_indexes (:py:class:'json') - A json with the {period:gain_index} for all the tested periods.        
    """
    # Define gain_index and period
    gain_index = float('-inf')
    p = -1  # best period
    periods_gain_indexes = {}
    m = model

    # Apply z-normalisation to the time series
    ts = zscore(ts)

    # Check if dates are provided and create a df
    if dates != 'None':
        # Create df
        d = {'DATE': dates, 'values': ts}
        ts = pd.DataFrame(d)

        # create datetime index passing the datetime series
        datetime_index = ts['DATE']

        # Complete the call to convert the date column
        datetime_index = pd.to_datetime(datetime_index, format='%m/%d/%Y')
        ts = ts.set_index(datetime_index)
        # we don't need the column anymore
        ts.drop('DATE', axis=1, inplace=True)
        ts = ts['values'].tolist()

    # Find the period with the smallest mean_abs error
    for i in periods:

        # Seasonal decomposition
        result = seasonal_decompose(ts, period=i, model=m)

        trend = np.where(np.isnan(result.trend), None, result.trend)
        seasonality = np.where(np.isnan(result.seasonal), None, result.seasonal)
        res_error = np.where(np.isnan(result.resid), None, result.resid)

        g = __get_gain_index(trend, seasonality, model=m, real_values=ts)

        if g > gain_index:
            gain_index = g
            p = i

        periods_gain_indexes.update({i: g})

    return p, periods_gain_indexes


def seasonal_decomposition(ts_df, date_column, data_column, periods, m):
    """ 
    This method performs the Triple Time Series Decomposition. The service takes as input a time series, the corresponding model type ("Multiplicative" or "Additive"), a list of periods parameters and the corresponding locale if applicable. The user can insert the path of the selected data or provide them in an array form. If the user provides more than 1 period parameters, the method selects the best one according to the best gain index. The provided time series is decomposed into three distinct components according to the selected model and period:
        
    - Trend: the increasing - decreasing value in the series.
    - Seasonality: the repeating short term cycle in the series.
    - Residual Error: the random variation in the series.

    An additive model suggests that the components are added toghether as follows:
    - y(t) = Trend + Seasonality + Residual Error

    While a multilicative model suggests that components are multiplied together as follows:
    - y(t) = Trend * Seasonality * Residual Error

    This implementation uses the "statsmodels.tsa.seasonal.seasonal_decompose" from the statsmodels library.
    
    :param ts_df: A Pandas dataframe containing the loaded time series.
    :type ts_df: pandas.DataFrame
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :param periods: A list containing the periods to be tested.
    :type periods: list
    :param m: The type of the model, 'additive' or 'multiplicative'.
    :type m: string
    :return:
        -result (:py:class:'statsmodels.tsa.seasonal.seasonal_decompose') - The result of the seasonal decomposition.
        -best_period (:py:class:'json') - The selected best period based on the minimum mean absolute value of the residual error component.
        -gain_indexes (:py:class:'json') - A json with the {period:gain_index} for all the tested periods.
        -fig1 (:py:class:'plotly.express') - A Potly figure containing the trend, seasonality and residual error components.
        -fig2 (:py:class:'plotly.express') - A Potly figure containing the seasonality component.
        """
    best_period, gain_indexes = __extract_best_period(ts_df[data_column].tolist(), ts_df[date_column].tolist(), periods,
                                                      model=m)
    result = seasonal_decompose(ts_df[data_column].tolist(), period=best_period, model=m)

    df = pd.DataFrame(
        {'Date': ts_df[date_column].tolist(),
         'Trend': result.trend,
         'Seasonality': result.seasonal,
         'Residual': result.resid,
         })
    df.set_index('Date')

    df.index = pd.to_datetime(df.index)

    print(df.columns[0:2])
    fig1 = px.line(df, x="Date", y=df.columns[1:4])
    fig2 = px.line(df.head(best_period), x=[str(i) for i in list(range(1, best_period + 1))], y="Seasonality",
                   title='Seasonality')

    # return result, best_period, gain_indexes, fig1, fig2, fig3
    return result, best_period, gain_indexes, fig1, fig2


def __rate_change(data, changepoints):
    """
    This method takes as input the data and the change points and calculates the absolute and directional rate change. Rate change is a metric that expresses the significance of a changing point. In order to calculate it we take the absolute percentage difference between the average values of two consecutive segments given a selected window.
    
    :param data: A list containing the time series values.
    :type data: list
    :param changepoints: The discovered change points.
    :type changepoints: list
    :return:
        -rate_changes (:py:class:'dict') - The change points along with their absolute rate change.
        -dir_rate_changes (:py:class:'dict') - The change points along with their direcitonal rate change.
    """

    # Initialise a dictionary that holds the changing points and their rate of change
    rate_changes = {}
    dir_rate_changes = {}

    # Convert the data into pandas format
    df = pd.DataFrame(data, columns=['values'])

    for i in range(len(changepoints) - 2):
        # Take the corresponding segments
        segment1 = df[changepoints[i]:changepoints[i + 1]]
        segment2 = df[changepoints[i + 1]:changepoints[i + 2]]

        # Calculate their average value
        mean1 = segment1.mean().values[0]
        mean2 = segment2.mean().values[0]

        # Calculate their absolute percentage difference

        abs_percentage_diff = abs((mean2 - mean1) / ((mean1 + mean2) / 2))

        dir_percentage_diff = ((mean2 - mean1) / ((mean1 + mean2) / 2))

        # Update rate_changes
        rate_changes[changepoints[i + 1]] = abs_percentage_diff
        dir_rate_changes[changepoints[i + 1]] = dir_percentage_diff

    return rate_changes, dir_rate_changes


def change_setection_single(ts_df, model, min_size):
    """
    This method identifies the change points within a single time series using the PELT approach (i.e., from the ruptures library: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/).
    
    :param ts_df: A Pandas dataframe containing the loaded time series
    :type ts_df: pandas.DataFrame
    :param model: The desired PELT model (can be either "l1", "l2", "normal", "rbf", "linear", or "ar"
    :type model: string
    :param min_size: The minimum distance (in number of timestamps) between two consecutive change points.
    :type min_size: int
    """
    signal = ts_df.iloc[:, 2:].values

    algo = rpt.Pelt(model=model, min_size=min_size, jump=5).fit(signal)
    my_bkps = algo.predict(pen=3)

    fig, (ax,) = rpt.display(signal, my_bkps, figsize=(16, 8))
    return fig


def change_detection_collective(ts_df_array, filenames, model, min_size, eps, min_samples, date_column, data_column):
    """
    This method identifies the change points within a collection of time series, ranks them and distinguishes them between global and local changes. In order to find the changing points, our implementation uses the PELT approach (from the ruptures library) and calculates their rate change metric. Subsequently, using DBSCAN, it creates some clusters which include the change points that are part of the same global change.
    
    :param ts_df_array: A list of Pandas dataframes containing the loaded time series.
    :type ts_df_array: list
    :param filenames: The corresponding filename of each time series.
    :type filenames: list
    :param model: The desired PELT model (can be either "l1", "l2", "normal", "rbf", "linear", or "ar".
    :type model: string
    :param min_size: Minimum number of samples between two change points (ruptures).
    :type min_size: int
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster (DBSCAN).
    :type eps: int
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself (DBSCAN).
    :type min_samples: int
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :return:
        -final_data (:py:class:'dict') - A dictionary containing the timestamps/dates of the identified change points, the name or id of the corresponding time series, the rate change of the change points, local-global cluster label (-1 stands for local changes).
        -data_clusters (:py:class:'dict') - A dictionary containing all the identified clusters (global changes), their aggregate and absolute aggregate rate changes and corresponding cluster scores, their starting and ending date or timestamp and the number of members of each cluster.
    """
    # Create Final Dataset
    final_data = pd.DataFrame(columns=['timestamps', 'rate_change', 'dir_rate_change', 'name'])
    num_of_stock = 1
    sftp_v = False

    # Iterate through the provided list with the stocks
    for i in range(0, len(ts_df_array)):

        values = ts_df_array[i][data_column].tolist()  # closing values
        stock_dates = ts_df_array[i][date_column].tolist()  # dates of the specific stock
        d = values
        entry = num_of_stock
        num_of_stock = filenames[i]

        d = np.array(d, dtype=np.float32)

        # Set up parameters for ruptures
        # ------------------------------
        jump = 5
        signal = d

        # Fit the model
        # -------------
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
        my_bkps = algo.predict(pen=3)

        # Calculate the rate of changes
        # -----------------------------
        my_bkps.insert(0, 0)
        results, dir_results = __rate_change(signal, my_bkps)

        # Create the final dataset
        # -----------------------
        df = pd.DataFrame(list(results.items()), columns=['timestamps', 'rate_change'])
        dir_df = pd.DataFrame(list(dir_results.items()), columns=['timestamps', 'dir_rate_change'])
        dir_rate_values = dir_df['dir_rate_change'].tolist()
        df['dir_rate_change'] = dir_rate_values
        df['name'] = entry

        # Add dates if exist
        # ------------------        
        changepoints = df['timestamps'].tolist()
        changepoints_dates = []
        for j in changepoints:
            changepoints_dates.append(stock_dates[j])
        df.insert(2, 'date', changepoints_dates)
        final_data = pd.concat([final_data, df], ignore_index=True)

    # Sort the changing points according to rate change index
    # -------------------------------------------------------
    final_data = final_data.sort_values(by='rate_change', ascending=False, ignore_index=True)

    # DBSCAN for identifying Global - Local changes
    # ----------------------------------------------

    # Τake the extracted timestamps of the changing points from the whole collection
    timestamps = final_data['timestamps'].values

    X = timestamps.reshape(-1, 1)

    # Call the DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Add labels to the data
    final_data['cluster label'] = labels

    # Cluster datarame
    # ----------------

    # Create a new df for the cluster dataframe
    data_changes = copy.deepcopy(final_data)

    # convert timestamps column to floats
    data_changes['timestamps'] = data_changes['timestamps'].astype(float)

    # get mean rate_change
    data_clusters = data_changes.groupby('cluster label', as_index=False)['rate_change', 'dir_rate_change'].mean()

    # get min-max date of each cluster
    data_clusters['min_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].min()[
        'timestamps']

    data_clusters['max_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].max()[
        'timestamps']

    # add an extra column that will hold info regarding with the number of changing points at each cluster
    data_clusters['counts'] = data_clusters['cluster label'].map(data_changes["cluster label"].value_counts())

    # add score column
    data_clusters['cluster_abs_score'] = data_clusters['rate_change'] * data_clusters['counts']

    # add  dir score column
    data_clusters['cluster_dir_score'] = data_clusters['dir_rate_change'] * data_clusters['counts']

    # Check if dates exist and add the min max date to the cluster dataframe
    if 'date' in data_changes.columns:
        min_dates = []
        max_dates = []
        for index, row in data_clusters.iterrows():
            min_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (
                    data_changes['timestamps'] == row['min_timestamp'])]['date'].reset_index(drop=True)[0])
            max_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (
                    data_changes['timestamps'] == row['max_timestamp'])]['date'].reset_index(drop=True)[0])

        data_clusters['min_dates'] = min_dates
        data_clusters['max_dates'] = max_dates

    # Remove the local changes cluster 
    data_clusters = data_clusters.iloc[1:, :]

    # Sort the df according to the score 
    data_clusters = data_clusters.sort_values(by=['cluster_abs_score'], ascending=False)
    data_clusters = data_clusters.reset_index(drop=True)

    return final_data, data_clusters


# -----------------------------------------
# P R E P R O C E S S I N G - F U N C T I O N S
# -----------------------------------------

def preproc_lowess_smoothing(ts_df_array, date_column=0, data_column=2, is_sorted=False, frac=0.025, it=0):
    """
    This method performs Locally Weighted Scatterplot Smoothing(LOWESS) on the list of pandas DataFrames.

    :param ts_df_array: A list of Pandas dataframes containing the loaded time series.
    :type ts_df_array: list
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :param is_sorted: This boolean is used in the lowess function, and it determines whether
    the data will be sorted by the x-values of the observed points before calculating lowess.
    :type is_sorted: bool
    :param frac: This float is used in the lowess function, and it determines the fraction of the data used when
    estimating each y-value. Between 0 and 1.
    :type frac: float
    :param it: This integer is used in the lowess function, and it determines the number of residual-based
    reweightings to perform.
    :type it: int
    :return:
        -ts_new_df_array (:py:class:'list') - A list containing a Pandas dataframe for each read time series where
        the values are replaced by their smoothed estimates.
    """
    ts_new_df_array = []

    for df in ts_df_array:
        new_df = pd.DataFrame()
        temp_df = df.copy()
        dates_float = range(len(df))
        filtered = lowess(temp_df.iloc[:, data_column], dates_float, is_sorted=is_sorted, frac=frac, it=it)
        new_values = filtered[:, 1]
        new_df['date'] = temp_df[date_column]
        new_df['time'] = '00:00'
        new_df['value'] = new_values
        new_df['normalized'] = zscore(new_df['value'])

        ts_new_df_array.append(new_df)

    return ts_new_df_array


def paa_aligned(ts_df_array, start_date, end_date, word_size=10, date_column=0, data_column=2, date_format='%m/%d/%Y'):
    """
    This method performs Piecewise Aggregate Approximation (PAA) on the list of Pandas dataframes.
    For each Pandas dataframe it reduces the dimension to word_size discrete levels.

    :param ts_df_array: A list of Pandas dataframes containing the loaded time series.
    :type ts_df_array: list
    :param start_date: The starting date for each time series Pandas dataframe.
    :type start_date: string
    :param end_date: The ending date for each time series Pandas dataframe.
    :type end_date: string
    :param word_size: The SAX word length.
    :type word_size: int
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type date_column: int
    :param date_format: The format of the date.
    :type date_format: string
    :return:
        -ts_new_df_array (:py:class:'list') - A list containing a Pandas dataframe for each read time series with
        word_size rows each.
    """
    ts_new_df_array = []
    new_times = []

    sax = __SAX(wordSize=word_size)
    begin = datetime.strptime(start_date, date_format).date()
    end = datetime.strptime(end_date, date_format).date()

    for df in ts_df_array:
        new_df = pd.DataFrame()
        temp_df = df.copy()
        temp_df[date_column] = pd.to_datetime(temp_df[date_column], format=date_format).dt.date
        mask = (temp_df[date_column] >= begin) & (temp_df[0] <= end)
        temp_df = temp_df.loc[mask]
        old_values = temp_df.iloc[:, data_column].values
        new_values, ds = sax.to_PAA(old_values)
        if len(new_times) == 0:
            for pos in range(len(ds)):
                new_times.append(df[0][int(math.ceil(np.mean(ds[pos])))])
        new_df['date'] = new_times
        new_df['time'] = '00:00'
        new_df['value'] = new_values
        new_df['normalized'] = zscore(new_df['value'])

        ts_new_df_array.append(new_df)

    return ts_new_df_array


# noinspection PyTypeChecker
# -----------------------------------------
# M I S S I N G  V A L U E S - F U N C T I O N S
# -----------------------------------------
def ts_df_array_to_df(ts_df_array, filenames):
    """
    This method creates a single pandas DataFrame with each column corresponding the 'value' column of
    a pandas DataFrame from the list of Pandas dataframes that contains the loaded time series.

    :param ts_df_array: A list of Pandas dataframes containing the loaded time series
    :type ts_df_array: list
    :param filenames: A list containing all read filenames
    :type filenames: list
    :return:
        -times (:py:class:'list') - A list of the dates corresponding to each row of the pandas DataFrame and
        is the same for all time series.
        -organized_df (pandas.DataFrame) - A pandas DataFrame with each column corresponding the 'value' column of
        a pandas DataFrame from the input list.
    """
    organized_df = pd.DataFrame()
    times = ts_df_array[0][0]
    organized_df['time'] = times
    for ts in range(len(ts_df_array)):
        organized_df[filenames[ts]] = ts_df_array[ts]['normalized']
    organized_df['time'] = pd.to_datetime(organized_df['time'])
    return times, organized_df


# -----------------------------------------
# ------ INTRODUCE MISSING VALUES -----
# -----------------------------------------
# ------ SINGLE COLUMN -----
# -----------------------------------------
def miss_perc(time_series: pd.DataFrame, miss_perc):
    """
    This method introduces missing values, on a pandas DataFrame, on the 1st time series column where the position
    of the first missing value is at 5% of 1st series from the top and the size of a single block varies based on the
    given miss_perc value.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param miss_perc: A float between 0 and 1 which determines the size of the missing values block.
    :type miss_perc: float
    :return:
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    # choose starting pos at 5% of the above chosen time series from the top
    top_five_perc = (5 * N) // 100

    # name of the chosen time series in the dataframe
    col = temp_time_series.columns[0]

    # maximum NaN pos in the chosen time series with starting point
    # the top_five_perc value to cover miss_perc of the series
    max_pos = int(miss_perc * N + top_five_perc)

    temp_time_series.loc[top_five_perc:max_pos - 1, col] = np.nan

    return temp_time_series


def ts_length(time_series: pd.DataFrame, length_perc):
    """
    This method introduces missing values, on a pandas DataFrame, on the 1st time series column where the position
    of the first missing value is at 5% of 1st series from the top, the length of the every column (time series) varies
    based on the given length_perc value and the size of the missing block is 10% of the chosen length of the time series.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param length_perc: A float between 0 and 1 which determines the length of the time series.
    :type length_perc: float
    :return:
        -time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the length of the every column varies based on the given length_perc value.
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.

    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    # change N based on length_perc
    N = int(length_perc * N)

    # choose starting pos at 5% of the time series from the top
    top_five_perc = (5 * N) // 100
    # start_pos_in_ts = random.randint(0, top_five_perc)

    # name of the chosen time series in the dataframe
    col = temp_time_series.columns[0]

    # maximum NaN pos or size of a single missing block is 10% * N with starting point the top_five_perc value
    max_pos = int(0.1 * N + top_five_perc)

    temp_time_series.loc[top_five_perc:max_pos - 1, col] = np.nan

    return time_series.loc[:N], temp_time_series.loc[:N]


def ts_nbr(time_series: pd.DataFrame, column_perc):
    """
    This method introduces missing values, on a pandas DataFrame, on the 1st time series column where the position
    of the first missing value is at 5% of 1st series from the top, the number of columns (time series) varies based on
    the given column_perc value and the size of the missing block is 10% of the length of the time series.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param column_perc: A float between 0 and 1 which determines the number of time series.
    :type column_perc: float
    :return:
        -time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the umber of columns varies based on the given column_perc value.
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    # change M based on column_perc
    M = int(column_perc * M)

    # name of the chosen time series in the dataframe
    col = temp_time_series.columns[0]

    # choose starting pos at 5% of the time series from the top
    top_five_perc = (5 * N) // 100

    # maximum NaN pos or size of a single missing block is 10% * N with starting point the top_five_perc value
    max_pos = int(0.1 * N + top_five_perc)

    temp_time_series.loc[top_five_perc:max_pos - 1, col] = np.nan

    return time_series.iloc[:, :M], temp_time_series.iloc[:, :M]


# -----------------------------------------
# ------ MULTIPLE COLUMNS -----
# -----------------------------------------
def miss_disj(time_series: pd.DataFrame):
    """
    This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
    The size of each missing block is length of time series/number of time series and the position of the first missing
    value in each time series is at column_index*(length of time series/number of time series).

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :return:
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    miss_size = N // M
    incr = 0
    for i in range(len(temp_time_series.columns)):
        col = temp_time_series.columns[i]
        temp_time_series.loc[incr:(((i + 1) * miss_size) - 1), col] = np.nan
        incr = ((i + 1) * miss_size)

    return temp_time_series


def miss_over(time_series: pd.DataFrame):
    """
    This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
    The size of each missing block is (2*length of time series)/number of time series for all columns except last
    where the size is length of time series/number of time series, and the position of the first missing value in each
    time series is at column_index*((length of time series/number of time series)/2).

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :return:
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    miss_size = (2 * N) // M
    incr = 0
    for i in range(len(temp_time_series.columns)):
        if i != (len(temp_time_series.columns) - 1):
            col = temp_time_series.columns[i]
            temp_time_series.loc[incr:((incr + miss_size) - 1), col] = np.nan
            incr = (((i + 1) * miss_size) / 2)
        else:
            col = temp_time_series.columns[i]
            temp_time_series.loc[incr:(((i + 1) * (miss_size / 2)) - 1), col] = np.nan
            incr = (((i + 1) * miss_size) / 2)

    return temp_time_series


# Missing data: 10 to 100% time series are incomplete;
# Missing blocks - size = 10, removed from a random series at a random position until
# a total of 0.1*N of all points of time series are missing.
def mcar(time_series: pd.DataFrame, active_perc):
    """
    This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
    The number of columns (time series) that have missing values varies based on the given active_perc value.
    For each chosen column a missing block of size = 10 is introduced at a random position until a total of
    (0.1*length of time series) of all points of the time series are missing.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param active_perc: A float between 0 and 1 that determines which columns (time series) will have missing values.
    :type active_perc: float
    :return:
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape
    r = random.Random(18931)

    active_cols = r.sample(range(0, M), int(M * active_perc))
    while len(active_cols) != 0:
        # random series
        col_choice = r.choice(active_cols)
        col = temp_time_series.columns[col_choice]
        # random position
        row = r.randint(0, N)
        # Missing blocks - size = 10
        temp_time_series.loc[row:row + 9, col] = np.nan
        if temp_time_series.loc[:, col].isna().sum() >= 0.1 * N:
            active_cols.remove(col_choice)

    return temp_time_series


# Missing data: 10 to 100 rows in each time series, position: at 5% of all series from the top.
def blackout(time_series: pd.DataFrame, m_rows):
    """
    This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
    We introduce missing blocks of size = m_rows on all the columns (time series) where the position
    of the first missing value is at 5% of all series from the top.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param m_rows: A int between 0 and 100 that determines the size of the missing blocks.
    :type m_rows: int
    :return:
        -temp_time_series (pandas.DataFrame) - A pandas DataFrame containing the loaded time series as columns and
        the missing values at the positions mentioned above.
    """
    temp_time_series = time_series.copy()
    random.seed(18931)
    # N = length of time series, M = number of time series
    N, M = temp_time_series.shape

    # choose starting pos at 5% of the time series from the top
    top_five_perc = (5 * N) // 100
    # start_pos_in_ts = random.randint(0, top_five_perc)

    # maximum NaN pos in the time series with starting point the top_five_perc value to cover m_rows of the series
    max_pos = int(m_rows + top_five_perc)

    temp_time_series.loc[top_five_perc:max_pos - 1, temp_time_series.columns] = np.nan

    return temp_time_series


def fill_missing_values(time_series: pd.DataFrame, algorithm, var_dict):
    """
    This is a missing value imputation method that uses a specific algorithm and its parameters to fill the missing
    values of a pandas Dataframe that has time series as columns.
    The available algorithms are divided in the following categories:
    1. Matrix Based
        1. SVD - Based
            1. IterativeSVD (SVDImpute)
            2. SoftImpute
            3. SVT
        2. Centroid Decomposition - Based
            1. CDMissingValueRecovery (CDRec)
        3. PCA - Based
            1. GROUSE
            2. SPIRIT
            3. ROSL
            4. PCA_MME
        4. Matrix Factorization - Based
            1. NMFMissingValueRecovery (TeNMF)
    2. Pattern Based
        1. OGDImpute
        2. TKCM
        3. DynaMMo
    3. Other
        1. MeanImpute
        2. LinearImpute
        3. ZaeroImpute

    :param time_series: A pandas DataFrame with missing values containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param algorithm: The name of the algorithm to be used for the imputation.
    :type algorithm: string
    :param var_dict: The parameters of the algorithm in a dictionary.
    :type var_dict: dict
    :return:
        -c_timeseries (pandas.DataFrame) - A pandas DataFrame with no missing values containing the loaded time series
        as columns.
    """
    c_timeseries = time_series.copy()
    lists = c_timeseries.values.tolist()
    m_ndarray = np.array(lists)

    if algorithm == 'OGDImpute':
        svd: np.ndarray = alg.OGDImpute.doOGDImpute(m_ndarray, var_dict['truncation'])
    elif algorithm == 'ROSL':
        svd: np.ndarray = alg.ROSL.doROSL(m_ndarray, var_dict['rank'], var_dict['reg'])
    elif algorithm == 'CDMissingValueRecovery':
        svd: np.ndarray = alg.CDMissingValueRecovery.doCDMissingValueRecovery(m_ndarray, var_dict['truncation'],
                                                                              var_dict['eps'])
    elif algorithm == 'GROUSE':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.GROUSE.doGROUSE(m_ndarray, var_dict['max_rank'])
    elif algorithm == 'DynaMMo':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.DynaMMo.doDynaMMo(m_ndarray, var_dict['H'], var_dict['max_iter'], var_dict['fast'])
    elif algorithm == 'IterativeSVD':
        svd: np.ndarray = alg.IterativeSVD.doIterativeSVD(m_ndarray, var_dict['rank'])
    elif algorithm == 'SoftImpute':
        svd: np.ndarray = alg.SoftImpute.doSoftImpute(m_ndarray, var_dict['max_rank'])
    elif algorithm == 'PCA_MME':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.PCA_MME.doPCA_MME(m_ndarray, var_dict['truncation'], var_dict['single_block'])
    elif algorithm == 'LinearImpute':
        svd: np.ndarray = alg.LinearImpute.doLinearImpute(m_ndarray)
    elif algorithm == 'MeanImpute':
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
    elif algorithm == 'NMFMissingValueRecovery':
        svd: np.ndarray = alg.NMFMissingValueRecovery.doNMFRecovery(m_ndarray, var_dict['truncation'])
    elif algorithm == 'SVT':
        svd: np.ndarray = alg.SVT.doSVT(m_ndarray, var_dict['tauScale'])
    elif algorithm == 'ZeroImpute':
        svd: np.ndarray = alg.ZeroImpute.doZeroImpute(m_ndarray)
    elif algorithm == 'Spirit':
        svd: np.ndarray = alg.Spirit.doSpirit(m_ndarray, var_dict['k0'], var_dict['w'], var_dict['lambda'])
    elif algorithm == 'TKCM':
        svd: np.ndarray = alg.TKCM.doTKCM(m_ndarray, var_dict['truncation'])
    else:
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)

    if algorithm == 'GROUSE' or algorithm == 'DynaMMo' or algorithm == 'PCA_MME':
        svd = np.transpose(svd)
        m_ndarray = np.transpose(m_ndarray)

    c_timeseries[c_timeseries.isnull()] = svd
    if c_timeseries.isnull().values.any():
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
        c_timeseries[c_timeseries.isnull()] = svd

    return c_timeseries


# -----------------------------------------
# T S F R E S H - F U N C T I O N S
# -----------------------------------------
def read_json_file_tsfresh(json_path: str):
    """
    Read the json file from the given path that contains the features to be calculated by tsfresh package.

    :param json_path: The path containing the json file.
    :type json_path: string
    :return:
        -json_decoded (dict) - A dictionary containing the tsfresh features.
    """
    with open(json_path, "r") as jf:
        json_decoded = json.load(jf)
    return json_decoded


def ts_fresh_json(df, json_decoded, no_time=False) -> pd.DataFrame:
    """
    This method uses tsfresh to calculate a comprehensive number of features.

    :param df: A pandas Dataframe with 3 columns (time, value,  id) or 2 columns (value,  id) as required to extract
    features from tsfresh.
    :type df: pandas.DataFrame
    :param json_decoded: A dictionary containing the tsfresh features.
    :type json_decoded: dict
    :param no_time: A boolean that if 'True' means that the 'time' column doesn't exist on the pandas Dataframe.
    :type no_time: bool
    :return:
        -tf (pandas.DataFrame) - A pandas DataFrame containing  the loaded time series as rows and the extracted
        features as columns.
    """
    if no_time:
        tf = extract_features(df, column_id="id",
                              column_value="value", default_fc_parameters=json_decoded, n_jobs=0, disable_progressbar=True)
    else:
        tf = extract_features(df, column_id="id", column_sort="time",
                              column_value="value", default_fc_parameters=json_decoded, n_jobs=0, disable_progressbar=True)

    return tf.rename(columns=lambda x: x.split("value__")[1]).rename(columns=lambda x: x.replace("_", " "))


# noinspection PyTypeChecker
def filter_important_features_tsfresh(df):
    """
    This method is used on the pandas Dataframe resulting from the tsfresh features extraction and filters the columns
    of said pandas Dataframe. The Anderson-Darling test is used in order to determine if the data of each feature are
    coming from a particular distribution. The features are being testing against normal, exponential, logistic and
    Gumbel (Extreme Value Type I) distributions. If the value of the Anderson-Darling test statistic is higher than the
    critical value for α = 0.025 against all distribution then we reject the null hypothesis of the test, and we have
    sufficient evidence to say the data of this particular feature do not follow any of these distributions and is
    excluded from the pandas Dataframe.

    :param df: A pandas DataFrame containing  the loaded time series as rows and the extracted features as columns.
    :type df: pandas.DataFrame
    :return:
        -new_df (pandas.DataFrame) - A pandas DataFrame containing  the loaded time series as rows and only the
        features that passed the above conditions as columns.
        -dist_df (pandas.DataFrame) - A pandas DataFrame containing in each row a feature and the first distribution
        that this feature didn't reject and therefore was not excluded from the resulting pandas Dataframe.
        .
    """
    # dataframe with 'important' features as columns
    new_df = pd.DataFrame(index=df.index)
    # dataframe with features and distributions
    dist_df = pd.DataFrame()
    # arrays to be used as columns in dist_df dataframe
    features_array = []
    dist_array = []

    for feature in df.columns:

        feature_array = df[feature].to_numpy(dtype=float)

        if is_not_finite(feature_array).any():
            feature_array = replace_missing_inf_values(feature_array)

        # print('\nFeature:', feature, '\n')
        if not is_not_finite(feature_array).any():
            for dist in ['norm', 'expon', 'logistic', 'gumbel', 'gumbel_r']:
                tmp = anderson(feature_array, dist=dist)

                # print('Statistic:', tmp.statistic)
                # print('Critical value at a significance level of 0.05 is:', tmp.critical_values[2],'\n')
                # noinspection PyTypeChecker
                if tmp.statistic < tmp.critical_values[2]:
                    new_df[feature] = feature_array
                    features_array.append(feature)
                    dist_array.append(dist)
                    break

    dist_df['feature'] = features_array
    dist_df['distribution'] = dist_array

    return new_df, dist_df


def replace_missing_inf_values(feature_array):
    """
    This method is used to replace the NaN , infinity and -infinity values of an array of numbers.
    The NaN is replaced by the mean of the numbers in the array, the infinity with mean + 3*std (standard deviation)
    and the -infinity with mean - 3*std.

    :param feature_array: An array that contains the values of a feature.
    :type feature_array: numpy.array
    :return:
        -feature_array (numpy.array) - A numpy array with no NaN, infinity and -infinity values.

    """
    feature_array_finite = feature_array[np.isfinite(feature_array)]
    mean_feature_array = np.nanmean(feature_array_finite)
    std_feature_array = np.nanstd(feature_array_finite)
    replace_pos_inf = mean_feature_array + 3 * std_feature_array
    replace_neg_inf = mean_feature_array - 3 * std_feature_array

    feature_array = np.nan_to_num(feature_array, copy=False, nan=mean_feature_array,
                                  posinf=replace_pos_inf, neginf=replace_neg_inf)
    return feature_array


def is_not_finite(arr):
    """
    This method returns an array of booleans that have 'True' in the positions where we do not have finite numbers.

    :param arr: An array of numbers.
    :type feature_array: numpy.array
    :return:
         -res (numpy.array) - A numpy array where 'True' if we have non-finite (NaN, infinity and -infinity) values.
    """
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res


# -----------------------------------------
# F O R E C A S T I N G - F U N C T I O N S
# -----------------------------------------

def forecast(data, start, end, forecast_range, period):
    """'forecast' method is responsible for forecasting time series data within
    the set start, end and forecast range.

    Parameters
    ----------
    data: list
        The provided list with the time series data
    start: int
        The start point of our predictions
    end: int
        The end point of our predictions
    forecast_range: int
        The number of points to predict at each step
    period: int
        The seasonal parameter

    Returns
    -------
    lists of lists:
        A list containing the predictions. Each prediction i in [start,end] is
        a list containing M = forecas_range values, which are the forecasted
        values starting from point i.
    int:
        The Best period parameter
    String:
        The  Best trend type
    String:
        The  Best seasonal type
    float:
        z-score of the best config


    """

    # Convert Data into pandas dataframe form
    d = pd.DataFrame(data)

    # Initialise the array that holds the predictions for each timestamp
    predictions = []

    # Initialise the arrays that holds the optimised parameters used for the
    # predicitons of each timestamp
    best_configurations = []
    seasonality = []
    trend_type = []
    seasonal_type = []
    z_scores = []

    # Set up the parameter space for Grid Search
    cfg_list = holt_winters_configs(seasonal=period)

    # Iterate through the selected timestamps
    for i in range(start, end):

        # Grid Search to tune the period - trend - seasonal parameters according to the performance
        # of the model on the near past. [ith time - forecast range --> ith time] This will be used to
        # find the best parameters that they will also be used for predicting
        # [ith --> ith + forecast range]

        best_config, z_score = grid_search_HWES(
            cfg_list, d.iloc[0:i - forecast_range], d.iloc[i - forecast_range:i], forecast_range)

        try:
            t, s, p = best_config
        except:
            continue

        # Train split
        df_train = d.iloc[0:i + 1]

        # Forecasting
        model = HWES(df_train, seasonal_periods=p, trend=t, seasonal=s)
        fitted = model.fit()
        forecasted_values = fitted.forecast(steps=forecast_range)

        # Append the data into the arrays
        predictions.append(forecasted_values.tolist())
        best_configurations.append([p, t, s])
        seasonality.append(p)
        trend_type.append(t)
        seasonal_type.append(s)
        z_scores.append(z_score)

    return predictions, seasonality, trend_type, seasonal_type, z_scores


# -----------------------------------------
# G R I D  S E A R C H -  O P T I M I S A T I O N  F U N C T I O N S
# ------------------------------------------------------------------

def holt_winters_configs(seasonal=[None]):
    """This method sets up the Holt Winter's parameter space for the Grid Search

    Parameters
    ----------
    data: list
        Seasonal values

    Returns
    -------
    list
        parameter space
    """
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    # create config instances
    for t in t_params:
        for s in s_params:
            for p in p_params:
                cfg = [t, s, p]
                models.append(cfg)
    return models


def grid_search_HWES(cfg_list, train, test, forecast_range):
    """This is the Grid Search method which is used to find the best parameter set for the
    Holt Winter algorithm for the near past in order to be used for our near future predictions

    Parameters
    ----------
    cfg_list: list
        Parameter space
    train: list
        Training data
    test: list
        Testing data to tune the performance of the model
    forecast_range: int
        The number of points to predict at each step

    Returns
    -------
    list:
        The best parameter set [trend, seasonal, period]
    float:
        The Z-score of the best config towards the other possible combinations
    """
    best_RMSE = np.inf
    best_config = []
    scores = []
    print(cfg_list)
    for i in range(len(cfg_list)):
        try:
            t, s, p = cfg_list[i]
            print('trend_type: ' + str(t))
            print('seasonal_type: ' + str(s))
            print('period: ' + str(p))
            model = HWES(train, seasonal_periods=p, trend=t, seasonal=s)
            fitted = model.fit()
            print(fitted)
            forecasted_values = fitted.forecast(steps=forecast_range)
            print(forecasted_values)
            rmse = np.sqrt(mean_squared_error(test, forecasted_values))

            print('rmse: ' + str(rmse))
            scores.append(rmse)
            if rmse < best_RMSE:
                best_RMSE = rmse
                best_config = cfg_list[i]
        except BaseException:
            continue

    # Z-score calculation
    mean = np.mean(scores)
    std = np.std(scores)
    z_score = (best_RMSE - mean) / std

    return best_config, z_score


def create_df_prophet(ts_df_array, choise_ts, date_column, data_column):
    """
    This method creates the pandas Dataframe to be used in prophet forecasting.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param choise_ts: An number corresponding to the position of a specific time series in the list.
    :type choise_ts: int
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :return:
         -df (pandas.Dataframe) - A pandas Dataframe in the format that is required to be used in  prophet forecasting.
    """
    df = ts_df_array[choise_ts]
    df = df.iloc[:, [date_column, data_column]]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    return df


def prophet_forecasting(ts_df_array, choise_ts, date_column, data_column, forecast_range,
                        yearly_seasonality='auto', weekly_seasonality='auto',
                        daily_seasonality='auto', seasonality_mode='additive'):
    """
    This method executes prophet forecasting for a specific time series and a specific foresting range.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param choise_ts: An number corresponding to the position of a specific time series in the list.
    :type choise_ts: int
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :param forecast_range: The range of forecasting in prophet.
    :type forecast_range: int
    :param yearly_seasonality: Fit yearly seasonality.
    Can be 'auto', True, False, or a number of Fourier terms to generate.
    :type yearly_seasonality: Any
    :param weekly_seasonality: Fit weekly seasonality.
    Can be 'auto', True, False, or a number of Fourier terms to generate.
    :type weekly_seasonality: Any
    :param daily_seasonality: Fit daily seasonality.
    Can be 'auto', True, False, or a number of Fourier terms to generate.
    :type daily_seasonality: Any
    :param seasonality_mode: 'additive' (default) or 'multiplicative'.
    :type seasonality_mode: string
    :return:
         -y_pred (numpy.array) - A numpy array containing the predictions for the forecasting range using prophet.
         -y_true (numpy.array) - A numpy array containing the true values corresponding to the forecasting range.
    """
    df = create_df_prophet(ts_df_array, choise_ts, date_column, data_column)
    train = df.drop(df.index[-forecast_range:])
    # define the model
    model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality, seasonality_mode=seasonality_mode)
    # fit the model
    model.fit(train)
    # define the period for which we want a prediction
    future = df['ds'].tail(forecast_range)
    future = pd.DataFrame(future)
    future['ds'] = pd.to_datetime(future['ds'])
    prophet_forecast = model.predict(future)
    # calculate MAE between expected and predicted values for december
    y_true = df['y'][-forecast_range:].values
    y_pred = prophet_forecast['yhat'].values

    return y_pred, y_true


def create_array_holtwinters(ts_df_array, choise_ts, date_column, data_column):
    """
    This method creates a pd.Series with the values of a specific time series to be used in Holt-Winters forecasting.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param choise_ts: An number corresponding to the position of a specific time series in the list.
    :type choise_ts: int
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :return:
         -ts_array (pd.Series) - A pd.Series containing the values of a specific time series.
         -datetime_index (pd.DatetimeIndex) - A pd.DatetimeIndex containing the corresponding datetime to the
         values of the pd.Series.
    """
    df = ts_df_array[choise_ts]
    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df.iloc[:, date_column])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df2 = df.set_index(datetime_index)
    ts_array = df2.iloc[:, data_column]

    return ts_array, datetime_index


