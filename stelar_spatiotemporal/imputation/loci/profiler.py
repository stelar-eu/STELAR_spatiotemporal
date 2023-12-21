import sys

sys.path.append('..')
from loci import time_series as ts
from pandas_profiling import ProfileReport
from pandas_profiling.model.typeset import ProfilingTypeSet
from pandas_profiling.config import Settings
from pandas_profiling.model.summarizer import PandasProfilingSummarizer
from pandas_profiling.report.presentation.core import Container
import geopandas as gp
import pandas as pd
import numpy as np
import algorithms as alg
from pandas_profiling.utils.paths import get_config
from loci.report import get_report_structure
import yaml

tsfresh_json_file = '../json_files/json_no_parameters.json'


# ------------------------------------#
# ------ PROFILER MAIN FUNCTION ------#
# ------------------------------------#
def profiler(df: pd.DataFrame, DataType: str, variables: dict):
    config = Settings()
    description = dict()
    if DataType == 'TimeSeries':
        config, description, _ = profiler_timeseries(df, variables['TimeSeries']['column'])
    elif DataType == 'Geospatial':
        config, description, _ = profiler_geospatial(df, variables['GeoSpatial'])

    return config, description


def profiler_timeseries(df: pd.DataFrame, time_column: str, mode: str = "default", minimal: bool = True):
    if minimal:
        config_file = get_config("config_minimal.yaml")

        with open(config_file) as f:
            data = yaml.safe_load(f)

        config: Settings = Settings().parse_obj(data)
    else:
        config: Settings = Settings()
    config.progress_bar = False
    sample_variables: Container = None
    description = None
    if mode == 'default' and len(df.columns) > 2:
        sample_time_series = df[df.columns[:4]]
        config, description, _ = profiler_timeseries(sample_time_series, time_column, "verbose", True)
        config.progress_bar = False
        report = get_report_structure(config, description)
        variables = report.content['body'].content['items'][1]
        item = variables.content['item'].content['items']
        sample_variables = Container(
            item,
            sequence_type="accordion",
            name="Sample TimeSeries",
            anchor_id="sample-timeseries-variables",
        )

        time_series_stacked = df.melt(id_vars=[time_column], value_vars=df.columns[1:],
                                      value_name='value', var_name='id')
        time_series_stacked = time_series_stacked.reindex(columns=[time_column, 'value', 'id'])

        time_series_stacked.rename(columns={time_column: 'time'}, inplace=True)

        time_series_stacked['time'] = pd.to_datetime(time_series_stacked['time']).apply(lambda x: x.value)

        # Missing values
        if np.isnan(time_series_stacked['value']).any():
            m_ndarray = np.array(time_series_stacked['value'])
            svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
            time_series_stacked['value'] = svd

        # Run tsfresh
        json_decoded = ts.read_json_file_tsfresh(tsfresh_json_file)
        ts_fresh_results = ts.ts_fresh_json(time_series_stacked, json_decoded, no_time=False)
        config.progress_bar = True
        profile = ProfileReport(ts_fresh_results, config=config, title="Profiling Report", minimal=minimal)
        description = profile.description_set
        description['table']['profiler_type'] = 'TimeSeries'
        description['analysis']['title'] = 'Profiling Report'
    elif mode == 'verbose' or len(df.columns) == 2:
        config.vars.timeseries.active = True
        config.progress_bar = True
        # if autocorrelation test passes then numeric timeseries else 'real' numeric
        config.vars.timeseries.autocorrelation = 0.3
        typeset = ProfilingTypeSet(config)
        custom_summarizer = PandasProfilingSummarizer(typeset)
        custom_summarizer.mapping['TimeSeries'].append(new_numeric_summary)
        profile = ProfileReport(df, tsmode=True, title="Profiling Report", sortby=time_column,
                                summarizer=custom_summarizer, config=config, progress_bar=True)
        description = profile.description_set
        description['table']['profiler_type'] = 'TimeSeries'

    return config, description, sample_variables


def profiler_geospatial(df: pd.DataFrame, longitude_column: str = None,
                        latitude_column: str = None, wkt_column: str = None, minimal: bool = True):
    if minimal:
        config_file = get_config("config_minimal.yaml")

        with open(config_file) as f:
            data = yaml.safe_load(f)

        config: Settings = Settings().parse_obj(data)
    else:
        config: Settings = Settings()

    if longitude_column is not None and latitude_column is not None:
        geom_lon_lat = "geometry_" + longitude_column + "_" + latitude_column
        s = gp.GeoSeries.from_xy(df[longitude_column], df[latitude_column])
        s = s.set_crs(4326, inplace=False, allow_override=True)
        df[geom_lon_lat] = s.to_wkt()

    profile = ProfileReport(df, config=config, progress_bar=True)
    description = profile.description_set
    description['table']['profiler_type'] = 'GeoSpatial'

    if wkt_column is not None:
        if not description['table']['types'].__contains__('Geometry'):
            description['table']['types']['Categorical'] -= 1
            description['table']['types']['Geometry'] = 1
        else:
            description['table']['types']['Categorical'] -= 1
            description['table']['types']['Geometry'] += 1

        s = gp.GeoSeries.from_wkt(df[wkt_column])
        s = s.set_crs(4326, inplace=False, allow_override=True)
        description['variables'][wkt_column]['type'] = 'Geometry'
        description['variables'][wkt_column]['mbr'] = s.total_bounds
        description['variables'][wkt_column]['union_convex_hull'] = s.unary_union.convex_hull.wkt
        description['variables'][wkt_column]['centroid'] = s.unary_union.centroid.wkt
        description['variables'][wkt_column]['length'] = s.unary_union.length
        missing = s.isna().tolist()
        if any(missing):
            description['variables'][wkt_column]['missing'] = True
            description['variables'][wkt_column]['n_missing'] = sum(missing)
            description['variables'][wkt_column]['p_missing'] = sum(missing) * 100 / len(missing)
        else:
            description['variables'][wkt_column]['missing'] = False
            description['variables'][wkt_column]['n_missing'] = 0
            description['variables'][wkt_column]['p_missing'] = 0.0
        description['variables'][wkt_column]['crs'] = s.crs

        count_geom_types = s.geom_type.value_counts()
        # number_of_types = count_geom_types.count()
        # description['variables'][wkt_column]['geom_types'] = {}
        # for i in range(number_of_types):
        #    description['variables'][wkt_column]['geom_types'][count_geom_types.index[i]] = int(count_geom_types[i])
        description['variables'][wkt_column]['geom_types'] = count_geom_types
    # https://gis.stackexchange.com/questions/359380/convex-hull-in-geopandas

    if longitude_column is not None and latitude_column is not None:
        if not description['table']['types'].__contains__('Geometry'):
            description['table']['types']['Categorical'] -= 1
            description['table']['types']['Geometry'] = 1
        else:
            description['table']['types']['Categorical'] -= 1
            description['table']['types']['Geometry'] += 1
        geom_lon_lat = "geometry_" + longitude_column + "_" + latitude_column
        description['variables'][geom_lon_lat]['type'] = 'Geometry'
        s = gp.GeoSeries.from_wkt(df[geom_lon_lat])
        s = s.set_crs(4326, inplace=False, allow_override=True)
        description['variables'][geom_lon_lat]['mbr'] = s.total_bounds
        description['variables'][geom_lon_lat]['union_convex_hull'] = s.unary_union.convex_hull.wkt
        description['variables'][geom_lon_lat]['centroid'] = s.unary_union.centroid.wkt
        description['variables'][geom_lon_lat]['length'] = s.unary_union.length
        missing = s.isna().tolist()
        if any(missing):
            description['variables'][geom_lon_lat]['missing'] = True
            description['variables'][geom_lon_lat]['n_missing'] = sum(missing)
            description['variables'][geom_lon_lat]['p_missing'] = sum(missing) * 100 / len(missing)
        else:
            description['variables'][geom_lon_lat]['missing'] = False
            description['variables'][geom_lon_lat]['n_missing'] = 0
            description['variables'][geom_lon_lat]['p_missing'] = 0.0
        description['variables'][geom_lon_lat]['crs'] = s.crs

        count_geom_types = s.geom_type.value_counts()
        # number_of_types = count_geom_types.count()
        # description['variables'][geom_lon_lat]['geom_types'] = {}
        # for i in range(number_of_types):
        #    description['variables'][geom_lon_lat]['geom_types'][count_geom_types.index[i]] = int(count_geom_types[i])
        description['variables'][geom_lon_lat]['geom_types'] = count_geom_types
    return config, description, None


def new_numeric_summary(config: Settings, series: pd.Series, summary: dict = None):
    if summary is None:
        summary = {}
    df = pd.DataFrame()
    dates_float = range(len(series))
    df['time'] = dates_float
    df['id'] = series.name
    df['value'] = series.values

    if np.isnan(df['value']).any():
        m_ndarray = np.array(df['value'])
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
        df['value'] = svd
    json_decoded = ts.read_json_file_tsfresh(tsfresh_json_file)
    ts_fresh_results = ts.ts_fresh_json(df, json_decoded, no_time=False)
    summary['tsfresh_features'] = ts_fresh_results.to_dict(orient='records')[0]
    return config, series, summary


def read_files(my_file, header=None, sep=','):
    try:
        df = pd.read_csv(my_file, header=header, sep=sep)
    except:
        return pd.DataFrame()

    return df
