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

ROOT_PATH='/gpfs/data1/vclgp/data/iss_gedi/h3/'

ROOT_PATH='/gpfs/data1/vclgp/xiongl/IS2global/data/h3/'

BAD_ORBITS=os.path.join(ROOT_PATH, 'api', 'issgedi_l4b_excluded_granules_r002_20230315a_plusManual.json')


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

def list_gedi_variables(keyword=None):
    """
    returns a dictionary of all columns available in the gedi-h3 files 
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
    
    returns h3 cells at resolution level 3 covered in the GEDI database
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

@dask.delayed
def h3_load_cell(h3_03, atl03=None, atl08=None, anci=None, pre_query=None):
    """
    h3_03: h3 level 3 index (string)
    atl03: ATL03 columns to load
    atl08: atl08
    
    returns dask delayed object with all columns merged in a single DataFrame of all shots within the h3_03 cell
    """
    loc = {i:j for i,j in locals().items() if i in ['atl03','atl08'] and j is not None}
    df = None
    base_path = ROOT_PATH + '{}/shots_hex/{}.parquet' ## {} place holder
    for prod,cols in loc.items():        
        if (len(loc) > 1 or anci is not None) and 'land_segments/delta_time' not in cols:
            cols.append('land_segments/delta_time')
        _path = base_path.format(prod, h3_03)  ## ...../h3/atl08/shots_hex/h3_03.parquet??? 

        # if read failed?
        #try:
        _ = pd.read_parquet(_path, engine='pyarrow', columns=cols)
        #except Exception as e:
        #    print(f'this file is missing or corrupt: {e}')
        #    continue # skip this file and loop to next.
        if df is None:
            df = _.reset_index()
            root_prod = prod
        else:
            df = df.merge(_, how='inner', on='land_segments/delta_time', suffixes=(f'_{root_prod}', f'_{prod}'))
    ###### if df is 20m segemtns, test failed here.
    # if any(df.columns.str.contains('000|001|002|003|004')):
    #    df = merge_20m_data(df)
    #    df = df.set_index('h3_12')
    #    print(df.head(1))
    #    sys.exit('test2')
    # else:
    df = df.set_index('h3_12') ### it is actually aready indexed by h3_12.
    
    if anci is not None:
        df = get_anci_file(df, anci)
    if pre_query is not None:
        df = df.query(pre_query)
    
    return df


#### i need to change this 

def load_gedi(atl03=None, atl08=None, anci=None, region=None, square_tiles=False, pre_query=None, ignore_parts=None):
    """
    atl03: tl03 columns to load
    atl08: atl08 columns to load
    region: geopandas.GeoDataFrame with boundaries from the area of interest
    square_tiles: load partitions as square tiles instead of hexagons
    
    returns a dask.DataFrame of all GEDI shots within the region (if any)
    """
    loc = {i:j for i,j in locals().items() if i in ['atl03','atl08'] and j is not None}
    
    if len(loc) == 0:
        raise ValueError('List variables to load from at least one ICESat-2 product level.')
    
    aoi = h3_aoi(region)
    # remove empty index in data folder.
    


    #print('aoi.index')
    #print(aoi.index)
    if ignore_parts is not None:
        aoi = aoi[~aoi.index.isin(ignore_parts)]
    
    if square_tiles:
        toi = egi.aoi_tiles(region)
        aoi = aoi.to_crs(toi.crs)
        th3 = toi.geometry.apply(lambda x: aoi.index[aoi.intersects(x)].to_list())
        l = th3.apply(len) 
        dfs = [egi_load_tile(i, j, atl03, atl08, anci, pre_query) for i,j in th3[l > 0].reset_index().to_numpy()]
    else:
        # skip not exsit cell 
        dfs=[]
        #for i in aoi.index:
        #   try:
        #      dfs.append(h3_load_cell(i, atl03, atl08, anci, pre_query))
        #   except Exception as e:
        #      print(f'file not exist: {e}')
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
    df: dask.DataFrame with h3-indexed GEDI data
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
    xras = make_geocube(xdf, resolution=res)
    
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
            xdf.to_hdf(out_path, key='GEDI', mode='w')
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

## -- post filtering methods

def list_bad_orbits(bad_orbs_path = BAD_ORBITS):
    with open(bad_orbs_path) as f:
        borbs = json.load(f)
        return borbs

def get_tile_id(x, y, tilesize=72):
    ease2_binsize = EASE_XY_RES[0]*tilesize, -EASE_XY_RES[1]*tilesize
    
    xid = int((x-EASE_XY_ORIGIN[0]) // ease2_binsize[0]) + 1
    yid = int((EASE_XY_ORIGIN[1]-y) // ease2_binsize[1]) + 1
    tile_id = f"X{xid:03d}Y{yid:03d}"
    
    return tile_id

def filter_bad_orbits(df, header = None):
    if len(df) == 0:
        return header
    borbs = list_bad_orbits()
    orb_gra = df.set_index('shot_number').root_file.str.extract(r'GEDI.*_O(?P<orbit>\d+)_0(?P<granule>\d)_.*')
    df = df.merge(orb_gra, left_on='shot_number', right_index=True)
    df['orbit_granule'] = df.apply(lambda x: int(f"{x.orbit}{x.granule}"), axis=1)

    pts = gpd.points_from_xy(df.lon_lowestmode, df.lat_lowestmode, crs=4326).to_crs(epsg=6933)
    df['tile_id'] = [get_tile_id(pt.x,pt.y) for pt in pts]
    
    def keep_shot(x):
        tid = x.tile_id.iloc[0]
        rm_orbs = borbs.get(tid)

        if rm_orbs is None:
            res = x.set_index('shot_number').orbit_granule.copy()
            res[:] = True
            return res

        res = ~x.set_index('shot_number').orbit_granule.isin(rm_orbs)
        return res

    keep = df.groupby('tile_id').apply(keep_shot)
    if type(keep) is pd.DataFrame:
        keep = keep.unstack()

    keep = keep.droplevel('tile_id')
    keep.name = 'include_granule_flag'

    df = df.merge(keep, left_on='shot_number', right_index=True)
    return df

def load_gedi_filtered(l2a=None, l2b=None, l4a=None, region=None, square_tiles=False, filter_degraded=True, filter_leaf_off=False, check_orbit_files=True, apply_filter=True, ignore_parts=None):
    """
    recipe: https://docs.google.com/presentation/d/1Z05XiaZGEX0cOjGj-mPQoBiMEaHn-bUUa-aIGAnoxiE/edit#slide=id.g146f6ca140a_0_6
    """
    
    l2a_vars  = ['shot_number', 'root_file', 'rx_assess/rx_maxamp', 'rx_assess/sd_corrected', 'geolocation/stale_return_flag', 'selected_algorithm', 'rx_assess/quality_flag', 'degrade_flag', 'surface_flag', 'lon_lowestmode', 'lat_lowestmode']
    l2a_vars += ['land_cover_data/pft_class','land_cover_data/region_class', 'land_cover_data/leaf_off_flag','land_cover_data/landsat_water_persistence', 'land_cover_data/urban_proportion']
    l2a_vars += ['geolocation/sensitivity_a1','geolocation/sensitivity_a2','geolocation/sensitivity_a5','geolocation/sensitivity_a10']
    l2a_vars += ['rx_processing_a1/zcross', 'rx_processing_a2/zcross', 'rx_processing_a5/zcross' ,'rx_processing_a10/zcross']
    l2a_vars += ['rx_processing_a1/toploc', 'rx_processing_a2/toploc', 'rx_processing_a5/toploc' ,'rx_processing_a10/toploc']
    l2a_vars += ['rx_processing_a1/rx_algrunflag', 'rx_processing_a2/rx_algrunflag', 'rx_processing_a5/rx_algrunflag' ,'rx_processing_a10/rx_algrunflag']
    
    out_cols = []
    if l4a is not None:
        out_cols += l4a
    if l2b is not None:
        out_cols += l2b
    if l2a is not None:
        out_cols += l2a
        l2a_vars = list(set(l2a_vars + l2a))
    out_cols = list(set(out_cols))
    
    gdf = load_gedi(l2a=l2a_vars, l2b=l2b, l4a=l4a, region=region, square_tiles=square_tiles, ignore_parts=ignore_parts)
       
    gdf['zcross'] = 0
    gdf['toploc'] = 0
    gdf['rx_algrunflag'] = 0
    gdf['sensitivity'] = 0
    
    for a in [1,2,5,10]:
        gdf['zcross']        += (gdf['selected_algorithm'] == a) * gdf[f"rx_processing_a{a}/zcross"]
        gdf['toploc']        += (gdf['selected_algorithm'] == a) * gdf[f"rx_processing_a{a}/toploc"]
        gdf['rx_algrunflag'] += (gdf['selected_algorithm'] == a) * gdf[f"rx_processing_a{a}/rx_algrunflag"]
        gdf['sensitivity']   += (gdf['selected_algorithm'] == a) * gdf[f'geolocation/sensitivity_a{a}']

    # algorithm run flag (used to decide if L2B/L4A algorithm are applied to a shot)
    gdf['algorithm_run_flag'] = (gdf['rx_assess/quality_flag'] == 1) & (gdf['rx_algrunflag'] == 1) & (gdf['zcross'] > 0) & (gdf['toploc'] > 0) & (gdf['sensitivity'] > 0) & (gdf['sensitivity'] < 1)
    
    # shots without high degradation of geolocation performance
    gdf['degrade_include_flag'] = ~(gdf['degrade_flag'] // 10).isin([5,7,8,9]) & ~(gdf['degrade_flag'] % 10).isin([1,2,4,5,6,7,9])

    # tropical forest flag (identify prediction strata with dense tropical forests)
    gdf['tropics_flag'] = gdf['land_cover_data/region_class'].isin([4,5,6]) & (gdf['land_cover_data/pft_class'] == 2)
    
    # land surface waveforms (non-urban). Note: urban/water are set to zero biomass in L4B, but excluded in L2B.
    gdf['land_surface_flag'] = (gdf['land_cover_data/landsat_water_persistence'] < 10) & (gdf['land_cover_data/urban_proportion'] < 50)

    gdf['quality_flag'] = gdf['algorithm_run_flag'] & gdf['land_surface_flag'] & (gdf['surface_flag'] == 1) & (gdf['geolocation/stale_return_flag'] == 0)
    gdf['quality_flag'] &= (gdf['rx_assess/rx_maxamp'] > (8 * gdf['rx_assess/sd_corrected']))
    gdf['quality_flag'] &= (gdf['geolocation/sensitivity_a2'] > .95)
    gdf['quality_flag'] &= ( ((gdf['geolocation/sensitivity_a2'] > .98) & gdf['tropics_flag']) | ~gdf['tropics_flag'] )
    
    if filter_degraded:
        gdf['quality_flag'] &= gdf['degrade_include_flag']
    if filter_leaf_off:
        gdf['quality_flag'] &= (gdf['land_cover_data/leaf_off_flag'] == 0)

    if check_orbit_files:
        meta = filter_bad_orbits(gdf.head())
        gdf = gdf.map_partitions(filter_bad_orbits, header = meta.head(0), meta = meta)
        gdf["quality_flag"] &= (gdf['include_granule_flag'])
    
    if apply_filter:
        gdf = gdf.query("quality_flag")
    else:
        out_cols += ['algorithm_run_flag','degrade_include_flag','tropics_flag','land_surface_flag','include_granule_flag','quality_flag']

    return gdf[out_cols]



########### merge 20 m products  need to update in the future.

@dask.delayed
def merge_20m(file_path=None, df_in=None, q = None):
    #### None = False
    if  file_path is not None:
        df = pd.read_parquet(file_path)

    else:
        df = df_in
    geo_cols = {'land_segments/canopy/h_canopy_20m_000': 'land_segments/canopy/h_canopy_20m',
                'land_segments/canopy/h_canopy_20m_001': 'land_segments/canopy/h_canopy_20m',
                'land_segments/canopy/h_canopy_20m_002': 'land_segments/canopy/h_canopy_20m',
                'land_segments/canopy/h_canopy_20m_003': 'land_segments/canopy/h_canopy_20m',
                'land_segments/canopy/h_canopy_20m_004': 'land_segments/canopy/h_canopy_20m',
                'land_segments/latitude_20m_000': 'land_segments/latitude_20m',
                'land_segments/latitude_20m_001': 'land_segments/latitude_20m',
                'land_segments/latitude_20m_002': 'land_segments/latitude_20m',
                'land_segments/latitude_20m_003': 'land_segments/latitude_20m',
                'land_segments/latitude_20m_004': 'land_segments/latitude_20m',
                'land_segments/longitude_20m_000': 'land_segments/longitude_20m',
                'land_segments/longitude_20m_001': 'land_segments/longitude_20m',
                'land_segments/longitude_20m_002': 'land_segments/longitude_20m',
                'land_segments/longitude_20m_003': 'land_segments/longitude_20m',
                'land_segments/longitude_20m_004': 'land_segments/longitude_20m',
                'land_segments/terrain/h_te_best_fit_20m_000': 'land_segments/terrain/h_te_best_fit_20m',
                'land_segments/terrain/h_te_best_fit_20m_001': 'land_segments/terrain/h_te_best_fit_20m',
                'land_segments/terrain/h_te_best_fit_20m_002': 'land_segments/terrain/h_te_best_fit_20m',
                'land_segments/terrain/h_te_best_fit_20m_003': 'land_segments/terrain/h_te_best_fit_20m',
                'land_segments/terrain/h_te_best_fit_20m_004': 'land_segments/terrain/h_te_best_fit_20m',
                'h3_12_20m_000': 'h3_12_20m',
                'h3_12_20m_001': 'h3_12_20m',
                'h3_12_20m_002': 'h3_12_20m',
                'h3_12_20m_003': 'h3_12_20m',
                'h3_12_20m_004': 'h3_12_20m'
               }

    ## filter first
    df_000 = df.drop(columns=df.columns[df.columns.str.contains('001|002|003|004')])

    df_000 = df_000.rename(columns=geo_cols)

###### data 001
    df_001 = df.drop(columns=df.columns[df.columns.str.contains('000|002|003|004')])
    df_001 = df_001.rename(columns=geo_cols) 
    

###### data 002
    df_002 = df.drop(columns=df.columns[df.columns.str.contains('000|001|003|004')])
    df_002 = df_002.rename(columns=geo_cols) 
    


###### data 003
    df_003 = df.drop(columns=df.columns[df.columns.str.contains('000|001|002|004')])
    df_003 = df_003.rename(columns=geo_cols) 
    


###### data 004
    df_004 = df.drop(columns=df.columns[df.columns.str.contains('000|001|002|003')])
    df_004 = df_004.rename(columns=geo_cols) 
    
    df_new =  pd.concat([df_000, df_001, df_002,df_003,df_004 ])
    df_new = df_new.set_index('h3_12_20m')
    ########## apply the filter
    if q is not None:
        df_new = df_new.query(q)
# Create a GeoDataFrame from the pandas DataFrame
    geometry = [Point(xy) for xy in zip(df_new['land_segments/longitude_20m'], df_new['land_segments/latitude_20m'])]
    gdf = gpd.GeoDataFrame(df_new , geometry=geometry,crs='EPSG:4326')
    if file_path is not None:
        # delete this file, later will write. 
        os.remove(file_path) # delete old file. 
        gdf.to_parquet(file_path, engine="pyarrow") ### directly write
    return gdf 

