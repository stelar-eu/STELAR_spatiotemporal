import os
import sys

sys.path.append('..')

import numpy as np
import pandas as pd
import altair as alt
import ipywidgets as widgets
import matplotlib.pyplot as plt
#import stumpy
from IPython.display import Markdown as md
from IPython.display import display
from matplotlib.patches import Rectangle
from loci import time_series as ts


# -----------------------------------------
# I P Y W I G D E T -  F U N C T I O N S
# ------------------------------------------------------------------
def missing_scenario_ipywidgets(time_series, list_scenarios):
    """
    This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose the scenario  which replaces values with NaN values in a pandas Dataframe resulting from the
    method ts_df_array_to_df.

    :param time_series: A pandas DataFrame containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param list_scenarios: A list of strings corresponding to each of the methods in the
    INTRODUCE MISSING VALUES section.
    :type list_scenarios: list
    :return:
         -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
         widgets tied to a function.
    """

    # read file
    def choose_scenario(scenario, perc) -> pd.DataFrame:
        perc = float(perc) / 100
        df_insert_nans = pd.DataFrame()
        temp_time_series = pd.DataFrame()
        if scenario == 'miss_perc':
            df_insert_nans = ts.miss_perc(time_series, perc)
        elif scenario == 'ts_length':
            temp_time_series, df_insert_nans = ts.ts_length(time_series, perc)
        elif scenario == 'ts_nbr':
            temp_time_series, df_insert_nans = ts.ts_nbr(time_series, perc)
        elif scenario == 'miss_disj':
            df_insert_nans = ts.miss_disj(time_series)
        elif scenario == 'miss_over':
            df_insert_nans = ts.miss_over(time_series)
        elif scenario == 'mcar':
            df_insert_nans = ts.mcar(time_series, perc)
        elif scenario == 'blackout':
            df_insert_nans = ts.blackout(time_series, perc * 100)

        display(df_insert_nans)

        if temp_time_series.empty:
            return time_series, df_insert_nans
        else:
            return temp_time_series, df_insert_nans

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=list_scenarios,
        value=list_scenarios[0],
        description='Choose File...',
        style=style
    )

    y = widgets.FloatSlider(description="Missing Percentage", min=0, step=10, max=100, value=30, style=style)

    x.set_title = 'Choose File...'

    # assign widgets to tabs
    tab_visualise = widgets.HBox([x, y])

    # create tabs
    tab_nest = widgets.Tab()

    # interact function in isolation
    f = widgets.interactive(choose_scenario, {'manual': True, 'manual_name': 'Read File'}, scenario=x, perc=y)
    tab_nest.children = [widgets.VBox(children=f.children)]
    tab_nest.set_title(0, 'Missing Scenario')
    display(tab_nest)

    return f


def algs_interactive_plot(df, new_df, alg_dict):
    """
    This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose the multiple algorithms for missing values imputation in a specific time series. An interactive
    altair plot is generated with the imputed values vs original/Real values.

    :param df: A pandas Dataframe containing the loaded time series as columns.
    :type df: pandas.DataFrame
    :param new_df: A pandas Dataframe containing the loaded time series as columns and has been filled, based on a
    specific scenario, with missing values.
    :type new_df: pandas.DataFrame
    :param alg_dict: A dictionary with the parameters needed for each of the algorithms to function.
    :type alg_dict: dict
    :return:
         -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
         widgets tied to a function.
    """
    algorithms = ['IterativeSVD', 'ROSL', 'CDMissingValueRecovery', 'OGDImpute',
                  'NMFMissingValueRecovery', 'MeanImpute', 'LinearImpute', 'ZeroImpute',
                  'DynaMMo', 'GROUSE', 'SoftImpute', 'PCA_MME', 'SVT', 'Spirit', 'TKCM']

    range_ = ['red', 'pink', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'brown',
              'lime', 'bisque', 'turquoise', 'navy', 'wheat', 'beige', 'lightsteelblue']

    original_ts = df.loc[:, new_df.isna().any()]
    new_missing_ts = new_df.loc[:, new_df.isna().any()]

    def plot_ts(choice_ts, choice_alg):
        fill_ts = pd.DataFrame()
        colors = []
        for alg in choice_alg:
            new_alg_df = ts.fill_missing_values(new_df, alg, alg_dict)
            fill_ts_alg = new_alg_df.loc[:, new_df.isna().any()]
            fill_ts[alg] = fill_ts_alg[choice_ts]
            colors.append(range_[algorithms.index(alg)])

        fill_ts[' original'] = original_ts[str(choice_ts)]
        colors.append('blueviolet')
        fill_ts[' original_with_nans'] = new_missing_ts[str(choice_ts)]
        colors.append('cadetblue')
        columns = fill_ts.columns
        df3 = fill_ts

        df3['Time'] = df3.index
        df3 = df3.melt(id_vars=['Time'], value_vars=columns,
                       value_name='Value', var_name='Algorithms')

        hover = alt.selection_single(
            fields=["Time"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(df3).mark_line(strokeWidth=4).encode(
                x="Time",
                y="Value",
                color=alt.Color('Algorithms', scale=alt.Scale(domain=columns.values, range=colors))
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        tooltips = (alt.Chart(df3, title="Plot Missing/Filled Timeseries").mark_rule().encode(
            x=alt.X('Time', type='quantitative', title='TimeStamp', axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('Value', title='Value'),
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Time", title="TimeStamp"),
                alt.Tooltip("Value", title="Value"),
                alt.Tooltip("Algorithms", title="Algorithms")
            ],
            color=alt.Color('Algorithms', scale=alt.Scale(domain=columns.values, range=colors)),
        ).add_selection(hover)
                    )

        plot = display((lines + points + tooltips).properties(
            width=650,
            height=400
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_title(
            fontSize=25
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=13).interactive())

        return plot

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=new_missing_ts.columns,
        value=new_missing_ts.columns[0],
        description='Choose ts...',
        style=style
    )

    multiselect = widgets.SelectMultiple(
        options=algorithms,
        value=['IterativeSVD'],
        rows=14,
        description='Algorithms',
        disabled=False
    )

    # create tabs
    tab_nest = widgets.Tab()

    # interact function in isolation
    f = widgets.interactive(plot_ts, {'manual': True, 'manual_name': 'View Plot'}, choice_ts=x, choice_alg=multiselect)
    controls = widgets.HBox(f.children[:1])
    button = widgets.HBox(f.children[2:3])
    output = f.children[-1]
    tab_nest.children = [widgets.VBox([controls, multiselect, button, output],
                                      layout=widgets.Layout(flex_flow='row wrap', justify_content='space-between'))]
    tab_nest.set_title(0, 'Fill Missing Values')
    display(tab_nest)

    return f


def algs_plus_lm_interactive_plot(df, new_df, pred_df, alg_dict, has_date=False):
    """
    This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose the multiple algorithms for missing values imputation in a specific time series. An interactive
    altair plot is generated with the imputed values vs original/Real values.

    :param df: A pandas Dataframe containing the loaded time series as columns.
    :type df: pandas.DataFrame
    :param new_df: A pandas Dataframe containing the loaded time series as columns and has been filled, based on a
    specific scenario, with missing values.
    :type new_df: pandas.DataFrame
    :param pred_df: A pandas Dataframe containing the loaded time series as columns and has been filled, based on the
    predictions of linear model.
    :type pred_df: pandas.DataFrame
    :param alg_dict: A dictionary with the parameters needed for each of the algorithms to function.
    :type alg_dict: dict
    :return:
         -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
         widgets tied to a function.
    """
    algorithms2 = ['IterativeSVD', 'ROSL', 'CDMissingValueRecovery', 'OGDImpute',
                   'NMFMissingValueRecovery', 'MeanImpute', 'LinearImpute', 'ZeroImpute',
                   'DynaMMo', 'GROUSE', 'SoftImpute', 'PCA_MME', 'SVT', 'Spirit', 'TKCM']

    range_ = ['red', 'pink', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'brown',
              'lime', 'bisque', 'turquoise', 'navy', 'wheat', 'beige', 'lightsteelblue']

    original_ts = df.loc[:, new_df.isna().any()]
    new_missing_ts = new_df.loc[:, new_df.isna().any()]

    def plot_ts(choice_ts, choice_alg):
        fill_ts = pd.DataFrame()
        colors = []
        for alg in choice_alg:
            new_alg_df = ts.fill_missing_values(new_df, alg, alg_dict)
            fill_ts_alg = new_alg_df.loc[:, new_df.isna().any()]
            fill_ts[alg] = fill_ts_alg[choice_ts]
            colors.append(range_[algorithms2.index(alg)])

        fill_ts[' original'] = original_ts[str(choice_ts)]
        colors.append('blueviolet')
        fill_ts[' lm_pred'] = pred_df[str(choice_ts)]
        colors.append('rosybrown')
        fill_ts[' original_with_nans'] = new_missing_ts[str(choice_ts)]
        colors.append('cadetblue')
        new_columns = fill_ts.columns
        df3 = fill_ts

        df3['Time'] = df3.index
        df3 = df3.melt(id_vars=['Time'], value_vars=new_columns,
                       value_name='Value', var_name='Algorithms')

        hover = alt.selection_single(
            fields=["Time"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(df3).mark_line(strokeWidth=4).encode(
                x="Time",
                y="Value",
                color=alt.Color('Algorithms', scale=alt.Scale(domain=new_columns.values, range=colors))
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        if has_date:
            tooltips = (alt.Chart(df3, title="Plot Missing/Filled Timeseries").mark_rule().encode(
                
                y=alt.Y('Value', title='Value'),
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("Time", title="TimeStamp"),
                    alt.Tooltip("Value", title="Value"),
                    alt.Tooltip("Algorithms", title="Algorithms")
                ],
                color=alt.Color('Algorithms', scale=alt.Scale(domain=new_columns.values, range=colors)),
            ).add_selection(hover)
                        )
        else:
            tooltips = (alt.Chart(df3, title="Plot Missing/Filled Timeseries").mark_rule().encode(
            x=alt.X('Time', type='quantitative', title='TimeStamp', axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('Value', title='Value'),
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Time", title="TimeStamp"),
                alt.Tooltip("Value", title="Value"),
                alt.Tooltip("Algorithms", title="Algorithms")
            ],
            color=alt.Color('Algorithms', scale=alt.Scale(domain=new_columns.values, range=colors)),
        ).add_selection(hover)
                    )

        plot = display((lines + points + tooltips).properties(
            width=650,
            height=400
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_title(
            fontSize=25
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=13).interactive())

        return plot

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=new_missing_ts.columns,
        value=new_missing_ts.columns[0],
        description='Choose ts...',
        style=style
    )

    multiselect = widgets.SelectMultiple(
        options=algorithms2,
        value=[],
        rows=14,
        description='Algorithms',
        disabled=False
    )

    # create tabs
    tab_nest = widgets.Tab()

    # interact function in isolation
    f = widgets.interactive(plot_ts, {'manual': True, 'manual_name': 'View Plot'}, choice_ts=x, choice_alg=multiselect)
    controls = widgets.HBox(f.children[:1])
    button = widgets.HBox(f.children[2:3])
    output = f.children[-1]
    tab_nest.children = [widgets.VBox([controls, multiselect, button, output],
                                      layout=widgets.Layout(flex_flow='row wrap', justify_content='space-between'))]
    tab_nest.set_title(0, 'Fill Missing Values')
    display(tab_nest)

    return f


# Seasonality
def plot_seasonality_ipywidgets(ts_df_array, filenames, periods, model, date_column=0, data_column=2):
    """
        This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose a specific time series and by executing the seasonal_decomposition method to present a Potly
    figure containing the trend, seasonality and residual error components and a Potly figure containing the seasonality
    component.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param filenames: The corresponding filename of each time series.
    :type filenames: list
    :param periods: A list containing the periods to be tested.
    :type periods: list
    :param model: The type of the model, 'additive' or 'multiplicative'.
    :type model: string
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :return:
        -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
        widgets tied to a function.
    """

    def plot_ts(choice_ts):
        pos = filenames.index(choice_ts)
        result, best_period, gain_indexes, fig1, fig2 = ts.seasonal_decomposition(ts_df_array[pos],
                                                                                  date_column,
                                                                                  data_column,
                                                                                  periods,
                                                                                  model)
        display(md("Best detected period: " + str(best_period)))
        display(md('#### Trend, seasonality and residual at the same plot'))
        display(fig1)
        display(fig2)

        return fig1, fig2

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=filenames,
        value=filenames[0],
        description='Choose ts...',
        style=style
    )

    # interact function in isolation
    f = widgets.interactive(plot_ts, {'manual': False, 'manual_name': 'View Plot'}, choice_ts=x)
    display(f)

    return f


# Forecasting
def forecasting_interactive_plot(ts_df_array, filenames, date_column, data_column, periods,
                                 yearly_seasonality='auto', weekly_seasonality='auto',
                                 daily_seasonality='auto', seasonality_mode='additive'):
    """
    This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose the multiple algorithms for forecasting in a specific time series and in specific forecasting
    range. An interactive altair plot is generated with the forecasting values vs original/Real values.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param filenames: The corresponding filename of each time series.
    :type filenames: list
    :param date_column: The column number containing the datetime of each entry in each file.
    :type date_column: int
    :param data_column: The column number containing the values in each file.
    :type data_column: int
    :param periods: A list containing the periods to be tested.
    :type periods: list
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
        -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
        widgets tied to a function.
    """
    forecast_algs = ['holtwinters', 'prophet']

    def plot_ts(choice_ts, forecast_range, choice_alg):
        pos = filenames.index(choice_ts)
        forecast_df = pd.DataFrame()
        colors = []
        array_without_actual = ts_df_array[pos].iloc[:-forecast_range, data_column].values.tolist()
        for alg in choice_alg:
            if alg == 'holtwinters':
                ts_array, datetime_index = ts.create_array_holtwinters(ts_df_array, pos, date_column, data_column)
                start = len(ts_array) - forecast_range
                end = len(ts_array) - forecast_range + 1
                blockPrint()
                pred = ts.forecast(ts_array.values, start, end, forecast_range, periods)
                enablePrint()
                pred_df = pd.DataFrame(pred[0][-1],
                                       index=datetime_index[(len(ts_array) - forecast_range):len(ts_array)])
                forecast_df['HoltWinters'] = array_without_actual + pred_df[0].values.tolist()
                colors.append('cadetblue')
            elif alg == 'prophet':
                blockPrint()
                forecast_df['Prophet'] = array_without_actual + ts.prophet_forecasting(ts_df_array, pos, date_column,
                                                                                       data_column, forecast_range,
                                                                                       yearly_seasonality,
                                                                                       weekly_seasonality,
                                                                                       daily_seasonality,
                                                                                       seasonality_mode)[0].tolist()
                enablePrint()
                colors.append('lime')
            else:
                forecast_df['Original'] = forecast_df['Actual']

        forecast_df['Actual'] = ts_df_array[pos].iloc[:, data_column]
        colors.append('blueviolet')
        columns = forecast_df.columns
        forecast_df['Time'] = forecast_df.index  # ts_df_array[pos].iloc[-forecast_range:, date_column]
        df3 = forecast_df

        df3 = df3.melt(id_vars=['Time'], value_vars=columns,
                       value_name='Value', var_name='Algorithms')

        hover = alt.selection_single(
            fields=["Time"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        lines = (
            alt.Chart(df3).mark_line(strokeWidth=4).encode(
                x="Time",
                y="Value",
                color=alt.Color('Algorithms', scale=alt.Scale(domain=columns.values, range=colors))
            )
        )

        # Draw points on the line, and highlight based on selection
        points = lines.transform_filter(hover).mark_circle(size=65)

        tooltips = (alt.Chart(df3, title="Forecasting").mark_rule().encode(
            x=alt.X('Time', type='quantitative', title='TimeStamp', axis=alt.Axis(labelAngle=-30)),
            y=alt.Y('Value', title='Value'),
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Time", title="TimeStamp"),
                alt.Tooltip("Value", title="Value"),
                alt.Tooltip("Algorithms", title="Algorithms")
            ],
            color=alt.Color('Algorithms', scale=alt.Scale(domain=columns.values, range=colors)),
        ).add_selection(hover)
                    )

        plot = display((lines + points + tooltips).properties(
            width=700,
            height=400
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_title(
            fontSize=30
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=13).interactive())

        return plot

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=filenames,
        value=filenames[0],
        description='Choose ts...',
        style=style
    )

    z = widgets.IntSlider(description="Forecasting Range", min=1, step=1,
                          max=len(ts_df_array[0]), value=30, style=style)

    multiselect = widgets.SelectMultiple(
        options=forecast_algs,
        value=['prophet'],
        rows=14,
        description='Algorithms',
        disabled=False
    )

    # create tabs
    tab_nest = widgets.Tab()

    # interact function in isolation
    f = widgets.interactive(plot_ts, {'manual': True, 'manual_name': 'View Plot'},
                            choice_ts=x, forecast_range=z, choice_alg=multiselect)
    ts_dropdown = widgets.HBox(f.children[:1])
    forecast_slider = widgets.HBox(f.children[1:2])
    button = widgets.HBox(f.children[3:4])
    output = f.children[-1]
    tab_nest.children = [widgets.VBox([ts_dropdown, forecast_slider, multiselect, button, output],
                                      layout=widgets.Layout(flex_flow='row wrap', justify_content='space-between'))]
    tab_nest.set_title(0, 'Forecasting')
    display(tab_nest)

    return f


# Anomaly Detection
def plot_anomaly_detection_ipywidgets(ts_df_array, filenames):
    """
    This method generates a VBox container containing a group of interactive widgets tied to a function that allows
    the user to choose a time series, a window size/sequence length and the number of anomalies to produce a Potly
    figure containing the time series, the corresponding to that window size matrix profile and the position of the
    anomalies.
    The STUMPY library is used compute the matrix profile of the time series , produced by the provided
    window size or sequence length, to find the subsequences located at the maximum points which are also referred to
    as a discords, novelties, or “potential anomalies”.

    :param ts_df_array: A list containing a Pandas dataframe for each read time series.
    :type ts_df_array: list
    :param filenames: The corresponding filename of each time series.
    :type filenames: list
    :return:
        -f (ipywidgets.widgets.interaction) - A VBox container containing a group of interactive
        widgets tied to a function.
    """

    # window size = m
    def plot_ts(choice_ts, m, anomalies):
        pos = filenames.index(choice_ts)
        df = ts_df_array[pos]
        mp = stumpy.stump(df.iloc[:, 2], m)
        ascending_mp = np.argsort(mp[:, 0])
        descending_mp = ascending_mp[::-1]
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 10))
        plt.suptitle('Discord (Anomaly/Novelty) Discovery', fontsize='35')
        plt.rc('font', size=20)
        axs[0].plot(df.iloc[:, 3].values)
        axs[0].set_ylabel(filenames[pos].split('.')[0], fontsize='25')

        for i in range(anomalies):
            rect = Rectangle((descending_mp[i], -4), m, 8, facecolor='lightgrey')
            axs[0].add_patch(rect)

        axs[1].set_xlabel('Time', fontsize='25')
        axs[1].set_ylabel('Matrix Profile', fontsize='25')
        for i in range(anomalies):
            axs[1].axvline(x=descending_mp[i], linestyle="dashed")

        axs[1].plot(mp[:, 0])
        return fig

    style = {'description_width': 'initial'}
    x = widgets.Dropdown(
        options=filenames,
        value=filenames[0],
        description='Choose ts...',
        style=style
    )

    y = widgets.IntSlider(description="Window Size", min=5, step=5, max=len(ts_df_array[0]), value=30, style=style)

    z = widgets.IntSlider(description="Number of potential anomalies", min=1, step=1, max=10, value=3, style=style)

    # create tabs
    tab_nest = widgets.Tab()

    # interact function in isolation
    f = widgets.interactive(plot_ts, {'manual': True, 'manual_name': 'View Plot'}, choice_ts=x, m=y, anomalies=z)
    tab_nest.children = [widgets.VBox(children=f.children)]
    tab_nest.set_title(0, 'Anomaly Detection')
    display(tab_nest)

    return f


# ------ Special Functions -----
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
