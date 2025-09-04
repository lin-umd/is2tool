# ICESat_2 h3 command tool
### *Author: Lin Xiong, Tiago de Conto*
### *Sep, 2025*

<br>

ICESat_2 h3 command tool for fast extraction of atl08 data at any region.

```
ih3_list_variables
ih3_list_resolutions
ih3_list_iso3
ih3_extract_shots
ih3_read_schema
ih3_aggregate
ih3_from_img
```

# Showcase
## Setting up CLI environment:
```
module unload gdal
conda activate /gpfs/data1/vclgp/xiongl/env/ih3
```
#
## `ih3_list_variables`

```
usage: ih3_list_variables [-h] [-a] [-g GREP] [-p PRODUCTS [PRODUCTS ...]]

List variables available in the ICESat_2 H3 database

options:
  -h, --help            show this help message and exit
  -a, --ancillary       include variables from ancillary database
  -g GREP, --grep GREP  match substring
  -p PRODUCTS [PRODUCTS ...], --products PRODUCTS [PRODUCTS ...]
                        search only specified ICESat_2 ATL08 and/or ancillary products
```

List variables available in the ICESat_2/H3 database as a 3 column table:
- *product*: product containing the varaible
- *column*: column corresponding to the variable in the ICESat_2/H3 product data frame
- *dtype*: data type

`ancillary` data were extracted from sources outside the ICESat_2 and matched to each ICESat_2 shot. Ancillaru data can be added upon request. Currently the following  products are available:
- [`glad_forest_loss`](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2021_v1_9): forest loss information from the Landsat derived product by Hansen et al (2012) at 30m resolution
- [`esa_land_cover`](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100): land cover classification from ESA/sentinel at 10m resolution
- [`nasa_dem`](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001): topography information from the NASA dem product at 30m resolution
- [`copernicus DEM`](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM): Copernicus DEM - Global and European Digital Elevation Model

### Example:

`ih3_list_variables -a -g slope`

```
| product    | column                              | dtype   |
|:-----------|:------------------------------------|:--------|
| atl08      | land_segments/terrain/terrain_slope | float   |
| copernicus | slope                               | int16   |
| nasadem    | slope                               | int16   |
```

#
## `ih3_list_resolutions`

```
usage: ih3_list_resolutions [-h] [-h3]

List resolution specs for H3 spatial indexing systems

optional arguments:
  -h, --help  show this help message and exit
  -h3, --h3   list H3 system only
```

List resolutions available for aggregation of ICESat_2 products using H3   spatial indexing systems.

H3 is the [hexagon hierarchical spatial index](https://h3geo.org/) developed by Uber. 


### Example:

`ih3_list_resolutions`

```
## -- H3 levels - hexagons
       average edge length (m)  average area (m^2)  number of unique indexes
level                                                                       
0                  1107712.591   4250546847700.000                       122
1                   418676.005    607220978242.900                       842
2                   158244.656     86745854034.700                      5882
3                    59810.858     12392264862.100                     41162
4                    22606.379      1770323551.700                    288122
5                     8544.408       252903364.500                   2016842
6                     3229.483        36129052.100                  14117882
7                     1220.630         5161293.200                  98825162
8                      461.355          737327.600                 691776122
9                      174.376          105332.500                4842432842
10                      65.908           15047.500               33897029882
11                      24.911            2149.600              237279209162
12                       9.416             307.100             1660954464122
13                       3.560              43.900            11626681248842
14                       1.349               6.300            81386768741882
15                       0.510               0.900           569707381193162

```

#
## `ih3_list_iso3`
```
usage: ih3_list_iso3 [-h] [-g GREP]

List iso3 country codes available in the postgres database

options:
  -h, --help            show this help message and exit
  -g GREP, --grep GREP  match substring in country names
```

Quick access to 3 letter country codes. Those countries boundarias are available in the cluster's postgres database and may be used to query ICESat_2 shots with `ih3_extract_shots`.

### Example:

`ih3_list_iso3 -g gabon`

```
|    | iso3   | name   |
|---:|:-------|:-------|
|  0 | GAB    | Gabon  |
```

#
## `ih3_extract_shots`

```
usage: ih3_extract_shots [-h] -o OUTPUT [-r REGION] [-atl08 ATL08 [ATL08 ...]] [-a ANCI] [-t0 TIME_START] [-t1 TIME_END]
                         [-q QUERY] [-q_20m QUERY_20M] [-b] [-m]  [-f FORMAT] [-n CORES] [-s THREADS] [-A RAM] [-p PORT]

Filter and export IS2 ATL08 data using h3

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory or file path
  -r REGION, --region REGION
                        path to vector (.shp, .gpkg, .kml etc.) or raster (.tif) file with region of interest to extract shots
                        from OR iso3 country code (if not, query all land hexs in data, do not query all!)
  -atl08 ATL08 [ATL08 ...], --atl08 ATL08 [ATL08 ...]
                        ICESat_2 atl08 variables to export
  -a ANCI, --anci ANCI  quoted dictionary of ancillary variables to export - e.g. "{'glad_forest_loss':['loss','lossyear']}"
  -t0 TIME_START, --time_start TIME_START
                        start date to filter shots [YYYY-MM-DD]
  -t1 TIME_END, --time_end TIME_END
                        end date to filter shots [YYYY-MM-DD]
  -q QUERY, --query QUERY
                        single string with custom filters upon listed variables - use pandas.DataFrame.query notation
  -q_20m QUERY_20M, --query_20m QUERY_20M
                        use only for 20m segment filtering [land_segments/canopy/h_canopy_20m,
                        land_segments/terrain/h_te_best_fit_20m]
  -b, --strong_beam     select only strong beams
  -m, --merge           merge outputs and export to single file
  -f FORMAT, --format FORMAT
                        output files format [default = parquet]
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 5]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 20]
  -p PORT, --port PORT  port where to open dask dashboard [default = 8787]
```

Extract spatially indexed ICESat_2 shots for a given region (or globally), filtered or not, geolocated or not, using distributed processes with RAM thresholding. 

If later aggregation of shots in large regions will be performed with `ih3_aggregate`, avoid passing the `-m, --merge` flag, since merged outputs are not processed in parallel.

You can follow the progress of your queries through the dask dashboard at the `-p, --port` adress. For example, in order to access it locally at the CLI's default port 10000 grom gsapp8, connect to the your desired GEOG cluster forwarding the desired port as in:

```
ssh -L 10000:localhost:10000 <username>@gsapp8
```

Then, once the `ih3` tool is running, navigate to `localhost:10000` in the browser of your local machine.

### Example:

Let's query canopy height from high quality geolocated ICESat_2 shots over Gabon and export the results to a directory called `is2_gabon`: 

`ih3_extract_shots -o /gpfs/data1/vclgp/xiongl/tmp/is2_gabon   -r GAB --atl08 land_segments/canopy/h_canopy \
-q '`land_segments/canopy/h_canopy` < 200' \
-t0 2023-01-01 -t1 2023-12-31  --strong_beam --cores 20 `

The output files stored at `is2_gabon` are spatially partitioned - i.e. each file contains ICESat_2 shots with the same parent hexagon at H3 resolution 3.

#
## `ih3_read_schema`

```
usage: ih3_read_schema [-h] -i INPUT

List variables available in the ICESat_2 H3 database

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        directory or file path [parquet or gpkg] - generated by `ih3_extract_shots`
```

Print the column names and data types of exported ICESat_2/H3 datasets.

### Example:

`ih3_read_schema -i is2_gabon`

```
|    | column                        | dtype   |
|---:|:------------------------------|:--------|
|  0 | land_segments/delta_time      | double  |
|  1 | land_segments/latitude        | float   |
|  2 | orbit_info/sc_orient          | int8    |
|  3 | land_segments/longitude       | float   |
|  4 | root_beam                     | string  |
|  5 | land_segments/canopy/h_canopy | float   |
|  6 | geometry                      | binary  |
|  7 | h3_12                         | string  |
```

#
## `ih3_aggregate`

```
usage: ih3_aggregate [-h] -i INPUT -o OUTPUT [-m MAPPER] [-r RES] [-d DROP_COLUMNS [DROP_COLUMNS ...]]
                     [-u USE_COLUMNS [USE_COLUMNS ...]] [-g] [-t] [-f FORMAT] [-n CORES] [-s THREADS] [-A RAM] [-p PORT]

Aggregate icesat-2 shots spatially using h3 (hexagons) system

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file or directory with indexed icesat-2 shots
  -o OUTPUT, --output OUTPUT
                        output directory or file name
  -m MAPPER, --mapper MAPPER
                        how to aggregate shots under pandas standard - string with single function name or quoted
                        list/dictionary: 'mean' or "['mean', 'std', 'count']" or "{'agbd':['mean','count'],
                        'sensitivity':['mean']}"
  -r RES, --res RES     resolution level of h3 system to aggregate to
  -d DROP_COLUMNS [DROP_COLUMNS ...], --drop_columns DROP_COLUMNS [DROP_COLUMNS ...]
                        columns to ignore in aggregation - may be useful for non-dictionary mappers
  -u USE_COLUMNS [USE_COLUMNS ...], --use_columns USE_COLUMNS [USE_COLUMNS ...]
                        only aggregate few columns
  -g, --img             force image output - resamples hexagons if index is h3
  -t, --tiles           export tiles instead of merging outputs to single vector/img file
  -f FORMAT, --format FORMAT
                        output files format [default = parquet]
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 10]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 20]
  -p PORT, --port PORT  port where to open dask dashboard [default = 8787]

```

Aggregate exported shots into vector [.gpkg] or raster [.tif] files at a given resolution. If rasterizing shots indexed with the H3 system, the output pixel size will be equivalent to the hexagons average diameter and their values will be resampled from overlapping hexagon centroids.

### Example:
`ih3_aggregate -i /gpfs/data1/vclgp/xiongl/tmp/is2_gabon -o /gpfs/data1/vclgp/xiongl/tmp/is2_gabon_r7 -m mean -r 7 -u land_segments/canopy/h_canopy`


#
## `ih3_from_img`

```
usage: ih3_from_img [-h] -o OUTPUT -i IMG [-f FORMAT] [-b BAND_NAMES [BAND_NAMES ...]]
                    [-w WINDOW_OPERATIONS [WINDOW_OPERATIONS ...]] [-y] [-e] [-m] [-g] [-l FILLNA] [-d] [-r] [-n CORES]
                    [-s THREADS] [-A RAM] [-p PORT]

Incorporate information from image data to icesat-2 shots

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory or file path
  -i IMG, --img IMG     path to raster file or directory with tiles to incorporate.
  -f FORMAT, --format FORMAT
                        image format to look for [default = tif]
  -b BAND_NAMES [BAND_NAMES ...], --band_names BAND_NAMES [BAND_NAMES ...]
                        (optional) band names to use in output
  -w WINDOW_OPERATIONS [WINDOW_OPERATIONS ...], --window_operations WINDOW_OPERATIONS [WINDOW_OPERATIONS ...]
                        (optional) moving window operations to apply to image bands: list of 3 integer numbers representing band
                        number (0 indexed), window size (1-9 pixels) and operation id (0 = sum, 1 = mean, 2 = median, 3 = mode),
                        respectively; e.g. -w 033 151 152
  -m, --merge           merge outputs and export to single file
  -g, --geo             export file as georreferenced points [.gpkg]
  -l FILLNA, --fillna FILLNA
                        value to replace NAs in the input images
  -d, --dropna          drop NAs before exporting
  -r, --resume          check for files in the output directory and ignore processing for existing files
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 10]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 20]
  -p PORT, --port PORT  port where to open dask dashboard [default = 8787]
```

Extract pixel information at ICESat_2 shot locations.

### Example:

  `ih3_from_img -o /gpfs/data1/vclgp/xiongl/tmp/gabon -i /gpfs/data1/vclgp/xiongl/tmp/treecover2010_gabon.tif -b treecover2010`

