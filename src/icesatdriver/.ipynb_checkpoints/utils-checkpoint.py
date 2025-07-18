import os, h5py, re, glob, psycopg2, yaml, warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import dask
import dask.dataframe as ddf
from icesatdriver.config import *   #### import config files in this folder

DB_CONNECTION = "host='gsapp11.clust.gshpc.umd.edu' dbname='gedicalval' user='gediuser' password='laser'"
def execute_query(q, geo=False): #### query what ??? 
    con = psycopg2.connect(DB_CONNECTION)
    # connection to a PostgreSQL database
    result = gpd.read_postgis(q, con) if geo else pd.read_sql_query(q, con)
    con.close()
    return result 