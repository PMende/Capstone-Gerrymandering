'''Cleans congressional election data and loads it into a PSQL table.

This script currently requires no user input, and should run if simply
invoked like 'python cong_data_to_psql.py' from the command line.
'''

from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import os

import psycopg2
import pandas as pd

def import_excel():
    '''Loads congressional data into a dictionary of pandas dataframes'''

    df_dict = pd.read_excel(
        'data/raw/2016_all/wi_2016_congress.xlsx',
        sheetname=list(range(1, 9))
    )

    return df_dict

def clean_dfs(df_dict):
    '''Cleans the DFs to prepare them for exporting to PSQL database'''

    for key in df_dict:
        _fix_nans(df_dict[key])

        # Include only top 2 vote getters - usually DEM/REP
        df_dict[key] = df_dict[key][[0, 1, 3, 4]]

        _set_column_names(df_dict[key])

        _eunsure_dem_and_rep(df_dict[key])

        df_dict[key].dropna(inplace=True)
        df_dict[key] = df_dict[key][
            df_dict[key]['WARDS'] != 'County Totals:'
        ]

        return None

def _fix_nans(df):
    '''Clears certain NaN values as described in the comments'''

    # This fills the county column appropriately
    df['WEC Canvass Reporting System'].fillna(method='pad', inplace=True)

    # These two are used simply to get the appropriate political party
    # in the first row of the DF
    df['Unnamed: 3'].fillna(method='bfill', inplace=True)
    df['Unnamed: 4'].fillna(method='bfill', inplace=True)

    return None

def _set_column_names(df):
    '''Sets vote total column names to be those of the associated party'''

    cols = list(df.iloc[0])
    cols[0] = 'COUNTY'
    cols[1] = 'WARDS'
    rename_dict = {
        'WEC Canvass Reporting System': cols[0],
        'Unnamed: 1': cols[1],
        'Unnamed: 3': cols[2],
        'Unnamed: 4': cols[3]
    }
    df.rename(columns=rename_dict, inplace=True)

    return None

def _eunsure_dem_and_rep(df):
    '''Ensures there are DEM/REP cols in df and drops IND/LIB cols'''

    for party in {'REP', 'DEM'}:
        if not party in df.columns:
            df[party] = 0
    for col_name in df.columns:
        if not col_name in {'REP', 'DEM', 'COUNTY', 'WARDS'}:
            df.drop(col_name, axis=1, inplace=True)

    return None
