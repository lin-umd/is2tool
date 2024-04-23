import os, h5py, re, glob, psycopg2, yaml, warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import dask
import dask.dataframe as ddf
from gedidriver.config import *   #### import config files in this folder

DB_CONNECTION = "host='gsapp11.clust.gshpc.umd.edu' dbname='gedicalval' user='gediuser' password='laser'"
def execute_query(q, geo=False): #### query what ??? 
    con = psycopg2.connect(DB_CONNECTION)
    # connection to a PostgreSQL database
    ## geo data frame or data frame ???
    result = gpd.read_postgis(q, con) if geo else pd.read_sql_query(q, con)
    con.close()
    return result

def h5_soc_path(file):#### from file get year , doy
    year = file.split('_')[2][:4]
    doy = file.split('_')[2][4:7]
    return os.path.join(SOC_DIR, year, doy, file) ### get full path of that file 

def traverse_datasets(h5_file, root=None): ####  ???
    def h5py_dataset_iterator(g, prefix=''): #g h5 file name 
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'            
            if root is not None and not path.startswith(f"/{root}"):
                continue
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item) 
            elif isinstance(item, h5py.Group): # test for group (go down) 
                yield from h5py_dataset_iterator(item, path) ### recursive search, get item name 

    for path, _ in h5py_dataset_iterator(h5_file): ### 
        yield path    
    
def dataset_info(hdf_file, root=None): 
    info_map = {'path':[], 'rows':[], 'cols':[], 'dtype': []}
    with h5py.File(hdf_file, 'r') as f:
        for dset in traverse_datasets(f, root):
            info_map['path'].append(dset)           
            info_map['dtype'].append(f[dset].dtype)

            xy = f[dset].shape
            x = xy[0]
            y = 1 if len(xy) == 1 else xy[1]            
            info_map['rows'].append(x)
            info_map['cols'].append(y)            
    return pd.DataFrame(info_map)  #### get all data path and va
# path 	rows 	cols 	dtype
#  	/ancillary_data/atlas_sdp_gps_epoch 	1 	1 	float64

def dataset_var(file, var):
    with h5py.File(file, 'r') as f:
        return f.get(var)[:]
    
def get_gedi_prod(soc_path, prod='atl08'):
    prod_ = f"{prod[1]}_{prod[2].upper()}"
    prod_glob = re.sub("ATL\\d{2}_(\d{14}_\d{8}_).*(\.h5)", f"{prod_}_\\1*\\2",soc_path)
    prod_file = pd.DataFrame({'gedi0':glob.glob(prod_glob)})
    prod_file['date'] = prod_file.gedi0.apply(lambda x: os.path.getmtime(x))
    prod_file = prod_file.sort_values('date')

    if len(prod_file) > 1:
        f_latest = prod_file.gedi0.iloc[-1]
        f_rest = prod_file.gedi0.iloc[:-1]
        if not f_latest.endswith('_10algs.h5'):
            f_base = f_latest[:-3]
            f_bool = f_rest.str.match(f_base)            
            if f_bool.any():
                return f_rest[f_bool].iloc[0]

    return prod_file.gedi0.iloc[-1] if len(prod_file) > 0 else None

####ATL08_[yyyymmdd][hhmmss]_[ttttccss]_[vvv_rr].h5

class GEDIFile:
    def __init__(self, file_path):
        self.parse_file(file_path)
    
    def __str__(self):
        return yaml.dump(self)
    
    def parse_file(self, f):
        self.full_name = os.path.basename(f)
        f_base = re.sub('\.h5$', '', self.full_name)
        fl = f_base.split('_')
        self.product = fl[0]
        #self.level = fl[1]
        self.date = dt.datetime.strptime(fl[1], '%Y%m%d%H%M%S')
        self.date_str = fl[1]
        self.doy_date_str = self.date.strftime('%Y%m%d')        
        self.julian_date_str = self.date.strftime('%Y%j')
        self.time_str = self.date.strftime('%H%M%S')
        self.orbit = int(fl[2][0:3]) #tttt
        self.cycle = int(fl[2][4:5]) #cc
        self.segment = int(fl[2][6:7])#ss
        self.version = int(fl[3])
        self.release= int(fl[4])
        self.all_algorithms = fl[-1] == '10algs'


class GEDIShot(GEDIFile): ### get shot number
    soc_dir_glob = '/gpfs/data1/vclgp/data/iss_gedi/soc/20*/*/'
    soc_dir_glob = '/gpfs/data1/vclgp/xiongl/IS2global/data/brazil/20*/*/'
    def __init__(self, shot, product='atl08', file_query=True):
        self.prod = product
        self.parse_shot(shot)
        if file_query:
            self.query_shot()
    
    def parse_shot(self, shot):
        self.shot = shot
        self.orbit = shot // 10000000000000
        self.track = shot // 100000000000
        self.beam = shot % 10000000000000 // 100000000000
        self.beam_str = f'BEAM{self.beam:04b}'
        self.power = self.beam > 3
        self.reserved  = shot % 1000000000000 // 1000000000 % 100
        self.sub_orbit = shot % 1000000000 // 100000000
        self.shot_index = shot % 100000000
        self.glob_query = self.soc_dir_glob + f"GEDI01_B_*_O{self.orbit:05d}_{self.sub_orbit:02d}_*.h5"
        self.sql_query = f"select distinct filename from gedi_data.orbit_h5_files where orbit = {self.orbit} and granule = {self.sub_orbit} and product = 'GEDI0{self.prod[-2]}' and level = '{self.prod[-1].upper()}'"
    
    def query_shot(self, sql=True):
        if sql:
            fl = execute_query(self.sql_query).to_numpy().flatten()
            fl = [h5_soc_path(f) for f in fl]
        else:
            fl = glob.glob(self.glob_query)
            
        for f in fl:
            fp = get_gedi_prod(f, self.prod)
            f_shots = dataset_var(fp, f'{self.beam_str}/shot_number')
            if self.shot in f_shots:
                self.soc_path = fp
                self.parse_file(fp)
                return

# def get_rh(f, beam, rh_str): ## get rh 
#     rh = int(rh_str[-2:])
#     return f[f'{beam}/rh'][:,rh]

def wf_pad(wave, n_bins=1420, cval=0): ## not interest
    bin_diff = len(wave) - n_bins
    if bin_diff == 0:
        return wave    

    if bin_diff < 0:
        wave = np.pad(wave, (0,abs(bin_diff)), 'constant', constant_values=(cval,cval))
        return wave
    
    if bin_diff > 0:
        return wave[:n_bins]

def extract_waves(h5_file, beam, idx, tx=False):### not interest
    init = 't' if tx else 'r'
    
    noises = h5_file[f'/{beam}/noise_mean_corrected'][:][idx]
    starts = h5_file[f'/{beam}/{init}x_sample_start_index'][:][idx] - 1
    counts = h5_file[f'/{beam}/{init}x_sample_count'][:][idx]
    ends = starts + counts
    wfs = h5_file[f'/{beam}/{init}xwaveform'][:]
    
    waves = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        noise = noises[i]        
        wf = wfs[start:end]
        wf = wf_pad(wf, cval=noise)
        waves.append(wf)
    return np.array(waves)


####   
def load_h5_min(fpath, columns, include_source=True):  # columns my data columns 
    f = h5py.File(fpath, 'r')                         #  variables under beam .....
    if 'land_segments/delta_time' not in columns:
        columns.append('land_segments/delta_time')   ## IS2 land_segments/delta_time
    full_df = []
    for k in f.keys():  #### keys should have 6 beams : gt123lr
        if not k.startswith('gt'): continue
        dfs = {}                                
        for j in columns:
            d = f[f"{k}/{j}"][:]
            
            if d.ndim == 2:
                for col in range(d.shape[-1]):# access the last element of the shape tuple or list
                    jj = f"{j}_{col:03d}"## format to 3 digits.
                    dfs[jj] = d[:,col]
            else:
                dfs[j] = d

        dfs = pd.DataFrame(dfs)  ## incldue all data       
        if include_source: 
            dfs['root_beam'] = k   ## beam name 
        full_df.append(dfs)        
    full_df = pd.concat(full_df)
    if include_source:
        full_df['root_file'] = os.path.basename(fpath)
    f.close()    
    return full_df.dropna().set_index('land_segments/delta_time')

def load_h5_merged(fpath, columns, other_prod=None, other_columns=None):
    a = load_h5_min(fpath, columns)

    if not bool(other_prod):
        return a

    def recurse_h5(df, i=0):        
        if i >= len(other_prod):
            return df
            
        if type(other_prod) == tuple:
            p = other_prod[i]
            oc = other_columns[i]
            i += 1
        else:
            p = other_prod
            oc = other_columns
            i = len(other_prod)
            
        path = get_gedi_prod(fpath, p)
        idf = load_h5_min(path, oc, False)
        df = df.join(idf,how='inner', rsuffix=f"_{p}")
        return recurse_h5(df, i)
    
    return recurse_h5(a)

@dask.delayed
def delay_h5(fpath, columns, other_prod=None, other_columns=None):
    return load_h5_merged(fpath, columns, other_prod, other_columns)

def get_gedi_ddf(file_paths, columns, other_prod=None, other_columns=None):
    delay_list = [delay_h5(f, columns, other_prod, other_columns) for f in file_paths]
    return ddf.from_delayed(delay_list)

def get_waveform_ddf(file_paths, shots=None, cols = ['shot_number','noise_mean_corrected','rx_sample_count','rxwaveform'], delay=False): # not interested
    f0 = file_paths[0]
    if f0 == os.path.basename(f0):
        file_paths = [h5_soc_path(i) for i in file_paths]
    l1b_file_paths = [get_gedi_prod(i, 'l1b') for i in file_paths]
    if delay:
        wdf = ddf.from_delayed([dask.delayed(load_h5_min)(f, cols, False, shots) for f in l1b_file_paths])
    else:
        wdf = pd.concat([load_h5_min(f, cols, False, shots) for f in l1b_file_paths])    
    return wdf
    