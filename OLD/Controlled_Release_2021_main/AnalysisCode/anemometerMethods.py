# imports
import datetime
import pytz
import math
import pandas as pd


def appendFlightFeatureMetStats_Stanford(matchedDF, metDF): #, dt
    """calculate met stats using minute leading up to flight feature and first minute of flight feature
    :param matchedDF = dataframe with passes matched to controlled releases.
    :param metDF = dataframe with met data (timeseries)
    :param dt = # of minutes before and after "Flight Feature Time (UTC)" to average met data"""
    for idx, row in matchedDF.iterrows():
        #t_delta = datetime.timedelta(seconds=dt)
        #start = row['Detection Time (UTC)']-t_delta
        #end = row['Detection Time (UTC)']
        start = row['cr_avg_start']
        end = row['cr_avg_end']

        stats = calcMetStats_Stanford(metDF, start, end)
        for key, value in stats.items():
            matchedDF.loc[idx, 'cr_' + key] = value

    return matchedDF


def calcMetStats_Stanford(df, start, end):
    """calculate min, max, mean, std of wind speed
    :param df = dataframe with met data (timeseries)
    :param start = datetime to start moving window
    :param end = datetime to stop moving window"""

    df = df[(df.index >= start) & (df.index < end)]
    windspeedMPS_min = df['Speed_MPS'].astype(float).min()
    windspeedMPS_max = df['Speed_MPS'].astype(float).max()
    windspeedMPS_mean = df['Speed_MPS'].astype(float).mean()
    windspeedMPS_std = df['Speed_MPS'].astype(float).std()


    stats = {'Sonic_mps_min': windspeedMPS_min,
             'Sonic_mps_max': windspeedMPS_max,
             'Sonic_mps_mean': windspeedMPS_mean,
             'Sonic_mps_std': windspeedMPS_std,
             'SonicHeight_mAGL': 10,
             }

    return stats
