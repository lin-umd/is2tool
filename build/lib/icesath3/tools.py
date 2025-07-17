import sys, os, glob, dask, warnings, pyarrow, warnings, h3, pyproj, json, fiona
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
import h3pandas
from shapely.geometry import Polygon, LineString, Point
import xarray as xar
from rioxarray import merge
from geocube.api.core import make_geocube
from dask import dataframe as ddf
from dask.distributed import progress
import dask_geopandas
import pyarrow.parquet as pq
import subprocess
import multiprocessing
from tqdm import tqdm
import shutil

#import easegridindex as egi  #### package folder ... 

ROOT_PATH='/gpfs/data1/vclgp/xiongl/IS2global/data/h3/'


EASE_XY_RES = (1000.895023349556141, -1000.895023349562052)
EASE_XY_ORIGIN = (-17367530.445161499083042, 7314540.830638599582016)
EASE_XY_WH = (34704, 14616)

def read_parquet_schema(path):
    """
    path: parquet file path
    
    returns a pandas.DataFrame with the parquet column structure
    """
    schema = pyarrow.parquet.read_schema(path, memory_map=True)
    schema = pd.DataFrame(({"column": name, "dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
    return schema

def read_geopackage_schema(path):
    """
    path: gpkg file path
    
    returns a pandas.DataFrame with the gpkg column structure
    """
    gpkg_file = fiona.open(path, driver='GPKG')
    schema = pd.DataFrame([{'column':i, 'dtype':j} for i,j in gpkg_file.schema.get('properties').items()])
    return schema

def list_icesat_variables(keyword=None):
    """
    returns a dictionary of all columns available in the icesat-h3 files 
    """
    #prods = ['l2a','l2b','l4a'] ### right now only atl08
    prods = ['atl08']
    df = pd.DataFrame()
    for p in prods:
        f = glob.glob(ROOT_PATH + f'{p}/shots_hex/*.parquet')[0]
        schema = read_parquet_schema(f).assign(product = p) #### col: product, p
        df = pd.concat([df, schema])
    if bool(keyword):
        df = df[df.column.str.match(f".*{keyword}.*")] 
    return df.set_index('product')

def list_anci_variables(keyword=None): ## landcover, nasa dem etc.
     """
     returns a dictionary of all columns available in the ancillary files 
     """
     anci_dir = os.path.join(ROOT_PATH, 'ancillary') ### where is this file created ?????
     prods = os.listdir(anci_dir)
     df = pd.DataFrame()
     for p in prods:
         f = glob.glob(anci_dir + f'/{p}/*.parquet')[0]
         schema = read_parquet_schema(f).assign(product = p)
         df = pd.concat([df, schema])
     if bool(keyword):
         df = df[df.column.str.match(f".*{keyword}.*")] 
     return df.set_index('product')

@dask.delayed  # a function should be executed lazily as a Dask task
def read_gpkg(path, cols):
    df = gpd.read_file(path, ignore_geometry=True) ##### read h3_12 ???
    idcol = 'egi01' if 'egi01' in df.columns else 'h3_12'  
    return df.set_index(idcol)[cols]

def fix_geo_signs(g):
    """
    g: shapely polygon
    
    returns polygon with standardized vertices
    """
    x,y = g.boundary.coords.xy
    x = np.array(x)
    y = np.array(y)
    is_ok = all(np.abs(x) < 170) or all(x > 0) or all(x < 0)
    if is_ok: return g
    x[x < 0] += 360 
    pol = Polygon(zip(x,y))
    
    anti_meridian = LineString(((180, -90), (180, 90)))
    buffered_line = anti_meridian.buffer(0.000000001)
    mpol = pol.difference(buffered_line)
    return mpol


def h3_aoi(region=None):
    """
    region: geopandas.GeoDataFrame with boundaries from the area of interest
    
    returns h3 cells at resolution level 3 covered in the icesat database
    """
    atl08 = ROOT_PATH + 'atl08/shots_hex/'
    files = glob.glob(atl08 + '*.parquet')
    h3_03_cells = pd.DataFrame(data=[os.path.basename(i).split('.')[0] for i in files], columns=['h3_03']).set_index('h3_03')
    h3_03_cells = h3_03_cells.h3.h3_to_geo_boundary()
    h3_03_cells.geometry = h3_03_cells.geometry.apply(fix_geo_signs)    
    if region is not None:
        if not bool(region.crs):
            raise ValueError('input region has no CRS')
        reg = region.to_crs(h3_03_cells.crs)
        is_in = h3_03_cells.geometry.apply(lambda x: reg.intersects(x))
        for i in range(is_in.shape[-1]):
            tmp = is_in.iloc[:,i]
            if i == 0:
                all_in = tmp
            else:
                all_in |= tmp
        h3_03_cells = h3_03_cells[all_in]
    return h3_03_cells # if region is not given, return all cells that have biomass footprint.
    # 13527 cell in total, this is too many. 
@dask.delayed
def h3_load_cell(h3_03, atl03=None, atl08=None, anci=None, pre_query=None):
    """
    h3_03: h3 level 3 index (string)
    atl03: ATL03 columns to load
    atl08: atl08
    
    returns dask delayed object with all columns merged in a single DataFrame of all shots within the h3_03 cell
    """
    #  retrun local symbol table as a dictionary
    loc = {i:j for i,j in locals().items() if i in ['atl03','atl08'] and j is not None}
    df = None
    base_path = ROOT_PATH + '{}/shots_hex/{}.parquet' ## {} place holder
    for prod,cols in loc.items():        
        if (len(loc) > 1 or anci is not None) and 'land_segments/delta_time' not in cols:
            cols.append('land_segments/delta_time')
        _path = base_path.format(prod, h3_03)  ## ...../h3/atl08/shots_hex/h3_03.parquet??? 
        _ = pd.read_parquet(_path, engine='pyarrow', columns=cols)
        if df is None:
            df = _.reset_index()
            root_prod = prod
        else:
            df = df.merge(_, how='inner', on='land_segments/delta_time', suffixes=(f'_{root_prod}', f'_{prod}'))
    df = df.set_index('h3_12') ### it is actually aready indexed by h3_12.
    
    if anci is not None:
        df = get_anci_file(df, anci)
    if pre_query is not None:
        df = df.query(pre_query)
    
    return df




def load_is2(atl03=None, atl08=None, anci=None, region=None, pre_query=None, ignore_parts=None):
    """
    atl03: tl03 columns to load
    atl08: atl08 columns to load
    region: geopandas.GeoDataFrame with boundaries from the area of interest

    
    returns a dask.DataFrame of all icesat shots within the region (if any)
    """
    loc = {i:j for i,j in locals().items() if i in ['atl03','atl08'] and j is not None}
    
    if len(loc) == 0:
        raise ValueError('List variables to load from at least one ICESat-2 product level.')
    
    aoi = h3_aoi(region)


    if ignore_parts is not None:
        aoi = aoi[~aoi.index.isin(ignore_parts)]

    if len(aoi) == 0: 
        print('## -- no hex in roi!')
    print('## -- length of h3 cells ', len(aoi))

    dfs=[]
    dfs = [h3_load_cell(i, atl03, atl08, anci, pre_query) for i in aoi.index]
        
    
    return ddf.from_delayed(dfs)

def h3_parent(df, level, keep_child=False):
    """
    df: input h3-indexed dataframe
    level: desired cell level (from 3 to 12)
    keep_child: keep both input/output h3 levelsin the dataframe?
    
    returns dataframe upscaled to larger h3 cells
    
    --
    Reference on H3 resolution levels: https://h3geo.org/docs/core-library/restable/
    """
    import h3pandas
    if level < 3:
        warnings.warn('Not recommended grouping on levels < 3. Refer to dask.DataFrame.groupby for grouping shots across multiple files.')
    elif level > 12:
        raise ValueError('Not possible to aggregate shots on levels > 12. Use h3pandas to generate spatial indices from coordinates directly.')
    
    df_par = df.h3.h3_to_parent(level)       
    if keep_child:
        df_par = df_par.reset_index()
    
    df_par = df_par.set_index(f'h3_{level:02d}')    
    return df_par

def h3_aggregate(df, level, agg='mean', return_geometry=True, centroids=False, n_meta=10, nparts=1):
    """
    df: dask.DataFrame with h3-indexed icesat data
    level: h3 resolution level to aggregate into
    agg: aggregation mapping in any format accepeted by pandas.DataFrame.groupby
    return_geometry: generate geometry for each h3 hexagon?
    
    returns spatially aggregated (geo)dataframe per h3 cell at the specified resolution
    
    --
    Reference on H3 resolution levels: https://h3geo.org/docs/core-library/restable/
    """
    def fgeo(x):
        import h3pandas
        if type(x.columns) is pd.MultiIndex:
            x.columns = ['_'.join(i) for i in x.columns]
        
        if centroids:
            x = x.h3.h3_to_geo()
        else:
            x = x.h3.h3_to_geo_boundary()
            x.geometry = x.geometry.apply(fix_geo_signs)
        return x

    nparts = df.npartitions if df.npartitions < nparts else nparts
    par_meta = h3_parent(df.head(npartitions=nparts), level)
    df_par = df.map_partitions(h3_parent, level=level, meta=par_meta)
    
    if callable(agg):
        def _set_index(x):
            x.index = x.index.get_level_values(0)
            return x

        meta = agg(df_par.head(n=n_meta, npartitions=nparts))
        df_par = df_par.map_partitions(lambda x: pd.DataFrame(x.groupby(level=0, group_keys=True).apply(agg)), meta=meta)
        
        meta = _set_index(meta)
        df_par = df_par.map_partitions(_set_index, meta=meta)
    else:
        df_par = df_par.map_partitions(lambda x: x.groupby(level=0).agg(agg))

    if return_geometry:
        geo_meta = fgeo(df_par.head(npartitions=nparts))
        df_par = df_par.map_partitions(fgeo, meta=geo_meta)

    return df_par



def h3_to_raster(xdf, res): 
    if (len(xdf) == 0):
       print(xdf)
       return None
    h3id = h3.h3_to_parent(xdf.index[0], 3)
    xras = make_geocube(xdf, resolution=res) # 
    
    xras = xras.assign_attrs(h3_03_id=h3id)
    for i in list(xras.data_vars):
        xras[i] = xras[i].assign_attrs(h3_03_id=h3id)
    return xras

def raster_parts(df, nparts=1):
    if df.index.name.startswith('egi'):
        def rasterize(x):
            idx = egi.egi_to_parent(x.copy(), 12).index.value_counts().idxmax()
            xras = egi.geodf_to_raster(x)
            for i in list(xras.data_vars):
                xras[i] = xras[i].assign_attrs(egi12_id=idx)
            return pd.Series(xras)
    else:
        nparts = df.npartitions if df.npartitions < nparts else nparts
        xdf = df.head(npartitions=nparts)
        h3lv = h3.h3_get_resolution(xdf.index[0])
        res =  h3.edge_length(h3lv, 'm') * 2 # hex diameter in meters
        utm = pyproj.database.query_utm_crs_info(xdf.crs.name, pyproj.aoi.AreaOfInterest(*xdf.total_bounds))[0]
        xras = make_geocube(xdf.to_crs(epsg=utm.code), resolution=(-res,res)).rio.reproject('EPSG:4326')
        xres, yres = xras.rio.resolution()
        rasterize = lambda x: pd.Series(h3_to_raster(x, (yres, xres)))

    rasp = df.map_partitions(rasterize, meta=pd.Series(dtype=object))
    return rasp

def export_parts(df, out_dir, fmt=None):   
    if not os.path.isdir(out_dir): ## create dir
        os.makedirs(out_dir)

    def write_func(xdf, out_dir=out_dir, fmt=fmt): ###
        #print('len(xdf):-----------'+ str(len(xdf)))
        if len(xdf) == 0: 
            return ''

        if type(xdf.iloc[0]) is xar.DataArray:
            attrs = {}
            basename="foo"
            ak = xdf.iloc[0].attrs.keys() # iloc[0] first row of the DataFrame
            if 'h3_03_id' in ak:
                basename = str(xdf.iloc[0].attrs['h3_03_id'])
                attrs = {'h3_03_id':basename}
            elif 'egi12_id' in ak:
                basename= xdf.iloc[0].attrs['egi12_id']
                attrs = {'egi12_id':basename}
                basename = str(basename)
            
            basename += '.tif' if fmt is None else f'.{fmt}'
            out_path = os.path.join(out_dir, basename)
            ras = xar.merge(xdf).assign_attrs(**attrs)
            ras.rio.to_raster(out_path)
            return out_path
        #  first index.
        basename = xdf.index[0] ### what is base name here? h3_12 index: 8c82ab821da01ff
        # print('base name is here: ')
        # print(basename)
        # sys.exit('check base name')



        if type(basename) is str:
            basename = h3.h3_to_parent(basename, 3) ### h3_03 index base name.
        elif type(basename) is np.uint64:
            basename = egi.egi_to_parent(xdf.copy(), 12).index.value_counts().idxmax()
            basename = str(basename)

        if type(xdf) is gpd.GeoDataFrame:
            basename += '.gpkg' if fmt is None else f'.{fmt}'
            out_path = os.path.join(out_dir, basename)
            if fmt in ['parq', 'parquet', 'pq']:
                xdf.to_parquet(out_path)
            else:
                xdf.to_file(out_path)
            return out_path
    
        basename += '.parquet' if fmt is None else f'.{fmt}'
        out_path = os.path.join(out_dir, basename)
        ### I did not set format.
        if fmt == 'txt':
            xdf.to_csv(out_path, sep='\t')
        elif fmt == 'csv':
            xdf.to_csv(out_path)
        elif fmt == 'h5' or fmt == 'hdf5':
            xdf.to_hdf(out_path, key='icesat', mode='w')
        else:
            xdf.to_parquet(out_path) 
        return out_path                    
    #print('df:-----------???') ####
    #print(df.compute())

    #map_partitions
    #computing and parallel processing of large dataset
    #pd.Series  one-dimensional labeled array
    # In this case, pd.Series(str) creates an empty Series object with a dtype of str.
    # meta: An empty pd.DataFrame or pd.Series that matches the dtypes and column names of the output.
    return df.map_partitions(write_func, meta=pd.Series(str)) # 
    ## Here I assume one partition [all rows that have same h3_03 index.]
def compute_raster(df, show_progress=False):    
    n = df.npartitions
    
    if show_progress:
        df = df.persist()
        progress(df)
        
    rdf = df.compute()
    if n == 1:
        return xar.merge(rdf)
    rdf = rdf[rdf.apply(lambda x: 1 not in x.shape)]
    rdf = xar.merge([merge.merge_arrays(rdf.loc[i]) for i in rdf.index.unique()])
    return rdf
## -- ancillary datasets support

def rh_filter(df, keep, rh_list, algo_column='selected_algorithm'):
    rh_selected = [f"rh_{i:03d}" for i in rh_list]
    df_merge = []
    for a in df[algo_column].unique():
        a_cols = [f"geolocation/rh_a{a}_{i:03d}" for i in rh_list]
        tdf = df.query(f'{algo_column} == {a}')[keep + a_cols].rename(columns={i:j for i,j in zip(a_cols, rh_selected)})
        df_merge.append(tdf)
    
    df = pd.concat(df_merge)
    return df

def get_anci_file(df, mapper={'quality_flags': ['quality_flag']}, merge_how='inner', header=None):
    if len(df) == 0:
        return header
    
    idx = df.index.name
    if idx == 'h3_12':
        h3ids = [h3.h3_to_parent(df.index[0], 3)]
    elif idx == 'egi01':
        h3ids = list(df.set_index('h3_12').h3.h3_to_parent(3).h3_03.unique())
    
    cdf = []
    for h3id in h3ids:
        unmerged=True
        for i,j in mapper.items():
            if type(j) is str:
                j = [j]
            
            if 'land_segments/delta_time' not in j:
                j.append('land_segments/delta_time')
            
            ipath = os.path.join(ROOT_PATH, 'ancillary', i, h3id + '.parquet')  
            is_file = os.path.exists(ipath)          
            if not is_file:
                ipath = glob.glob(os.path.join(ROOT_PATH, 'ancillary', i, '*.parquet'))[0]                
            
            idf = pd.read_parquet(ipath, columns=j, engine='pyarrow')            
            if not is_file:
                idf=idf.head(0)            
            
            if unmerged:
                mdf = idf
                unmerged = False
            else:
                mdf = mdf.merge(idf, on='land_segments/delta_time', how=merge_how)
        cdf.append(mdf)
    
    cdf = pd.concat(cdf)
    df = df.reset_index().merge(cdf, on='land_segments/delta_time', how=merge_how).set_index(idx)
    return df
    
def load_anci(gdf, mapper={'quality_flags': ['quality_flag']}, merge_how='inner'):
    for i in range(gdf.npartitions):
        meta = get_anci_file(gdf.partitions[i].head(), mapper=mapper, merge_how=merge_how)
        if meta is not None: break    
    gdf = gdf.map_partitions(get_anci_file, mapper=mapper, merge_how=merge_how, header=meta.head(0), meta=meta)
    return gdf


def get_tile_id(x, y, tilesize=72):
    ease2_binsize = EASE_XY_RES[0]*tilesize, -EASE_XY_RES[1]*tilesize
    
    xid = int((x-EASE_XY_ORIGIN[0]) // ease2_binsize[0]) + 1
    yid = int((EASE_XY_ORIGIN[1]-y) // ease2_binsize[1]) + 1
    tile_id = f"X{xid:03d}Y{yid:03d}"
    
    return tile_id





########### merge 20 m products

def get_h3_20m(df, res=12, postfix='20m', lat_col='land_segments/latitude', lng_col='land_segments/longitude'):
    geo_cols = {lat_col: 'lat', lng_col: 'lng'}
    tmp = df.rename(columns=geo_cols).h3.geo_to_h3(res).h3.h3_to_parent(3)
    add = {f"h3_{res:02d}_{postfix}": tmp.index} ### h3_03 alreay added # , "h3_03": tmp.h3_03.values
    return df.assign(**add) # add h3_12_20m   colum. 


def merge_20m(file_path=None, q_20m = None):
    df = pd.read_parquet(file_path)
    if not df.columns.str.contains('001|002|003|004').any(): return
    df_000 = df.drop(columns=df.columns[df.columns.str.contains('001|002|003|004')])
    df_000 = df_000.rename(columns=lambda x: x[:-4] if x.endswith('_000') else x)
    df_001 = df.drop(columns=df.columns[df.columns.str.contains('000|002|003|004')])
    df_001 = df_001.rename(columns=lambda x: x[:-4] if x.endswith('_001') else x)
    df_002 = df.drop(columns=df.columns[df.columns.str.contains('000|001|003|004')])
    df_002 = df_002.rename(columns=lambda x: x[:-4] if x.endswith('_002') else x) 
    df_003 = df.drop(columns=df.columns[df.columns.str.contains('000|001|002|004')])
    df_003 = df_003.rename(columns=lambda x: x[:-4] if x.endswith('_003') else x) 
    df_004 = df.drop(columns=df.columns[df.columns.str.contains('000|001|002|003')])
    df_004 = df_004.rename(columns=lambda x: x[:-4] if x.endswith('_004') else x)
    df_20m =  pd.concat([df_000, df_001, df_002,df_003,df_004])
    ########## apply the filter
    if q_20m is not None:
        df_20m = df_20m.query(q_20m)
    # update h3_12 index
    df_20m = get_h3_20m(df_20m, res=12, postfix='20m', lat_col='land_segments/latitude_20m', lng_col='land_segments/longitude_20m')
    df_20m = df_20m_ddf.compute()
    df_20m.set_index('h3_12_20m', drop=True, inplace=True)
    df_20m.rename_axis("h3_12", inplace=True)
    # Create geometry column from latitude and longitude
    geometry = gpd.points_from_xy(df_20m['land_segments/longitude_20m'],df_20m['land_segments/latitude_20m'],crs="EPSG:4326")
    # Convert to GeoDataFrame
    df_20m = gpd.GeoDataFrame(df_20m, geometry=geometry, crs="EPSG:4326")
    # delete this file, later will write. 
    os.remove(file_path) # delete old file.
    df_20m.to_parquet(file_path, engine="pyarrow") ### directly write new file.
    #return df_20m 