# ICESat_2 h3 command tool
### *Author: Lin xiong, Tiago de Conto*
### *April 23, 2024*

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
                        search only specified ICESat_2 [l2a, l2b, l4a] and/or ancillary products
```

List variables available in the ICESat_2/H3 database as a 3 column table:
- *product*: product containing the varaible
- *column*: column corresponding to the variable in the ICESat_2/H3 product data frame
- *dtype*: data type

`ancillary` data were extracted from sources outside the ICESat_2 and matched to each ICESat_2 shot. Ancillaru data can be added upon request. Currently the following  products are available:
- [`glad_forest_loss`](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2021_v1_9): forest loss information from the Landsat derived product by Hansen et al (2012) at 30m resolution
- [`esa_land_cover`](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100): land cover classification from ESA/sentinel at 10m resolution
- [`nasa_dem`](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001): topography information from the NASA dem product at 30m resolution

### Example:

`ih3_list_variables -a -g canopy`

```
| product   | column                                    | dtype   |
|:----------|:------------------------------------------|:--------|
| atl08     | land_segments/canopy/h_canopy             | float   |
| atl08     | land_segments/canopy/h_canopy_20m_000     | float   |
| atl08     | land_segments/canopy/h_canopy_20m_001     | float   |
| atl08     | land_segments/canopy/h_canopy_20m_002     | float   |
| atl08     | land_segments/canopy/h_canopy_20m_003     | float   |
| atl08     | land_segments/canopy/h_canopy_20m_004     | float   |
| atl08     | land_segments/canopy/h_canopy_abs         | float   |
| atl08     | land_segments/canopy/h_canopy_quad        | float   |
| atl08     | land_segments/canopy/h_canopy_uncertainty | float   |
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

List resolutions available for aggregation of ICESat_2 products using H3 and EGI spatial indexing systems.

H3 is the [hexagon hierarchical spatial index](https://h3geo.org/) developed by Uber. EGI stands for Ease Grid Index - a custom spatial index system with perfectly nested squares based on the EASE-Grid 2.0 Global CRS (EPSG:6933) - developed by yours truly ;)


### Example:

`ih3_list_resolutions`

```
## -- H3 levels - hexagons
       average edge length (m)  average area (m^2)
level                                             
0                  1281256.011   4357449416078.392
1                   483056.839    609788441794.134
2                   182512.957     86801780398.997
3                    68979.222     12393434655.088
4                    26071.760      1770347654.491
5                     9854.091       252903858.182
6                     3724.533        36129062.164
7                     1406.476         5161293.360
8                      531.414          737327.598
9                      200.786          105332.513
10                      75.864           15047.502
11                      28.664            2149.643
12                      10.830             307.092

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

`ih3_list_iso3 -g states`

```
|    | iso3   | name                                 |
|---:|:-------|:-------------------------------------|
|  0 | FSM    | Micronesia, Federated States of      |
|  1 | UMI    | United States Minor Outlying Islands |
|  2 | USA    | United States                        |
|  3 | VIR    | United States Virgin Islands         |
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
                        from OR iso3 country code (if not, query all land hexs in data)
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
  -p PORT, --port PORT  port where to open dask dashboard [default = 34999]
```

Extract spatially indexed ICESat_2 shots for a given region (or globally), filtered or not, geolocated or not, using distributed processes with RAM thresholding. 

If later aggregation of shots in large regions will be performed with `ih3_aggregate`, avoid passing the `-m, --merge` flag, since merged outputs are not processed in parallel.

You can follow the progress of your queries through the dask dashboard at the `-p, --port` adress. For example, in order to access it locally at the CLI's default port 10000 grom gsapp8, connect to the your desired GEOG cluster forwarding the desired port as in:

```
ssh -L 10000:localhost:10000 <username>@gsapp8
```

Then, once the `ih3` tool is running, navigate to `localhost:10000` in the browser of your local machine.

### Example:

Let's query foliage height diversity, canopy cover, biomass and structural complexity from high quality geolocated ICESat_2 shots over puerto_rico and export the results to a directory called `ICESat_2_puerto_rico`: 

`ih3_extract_shots -o /gpfs/data1/vclgp/xiongl/IS2global/process/puerto_rico   -r PRI --atl08 land_segments/canopy/h_canopy_20m \
-q_20m '`land_segments/canopy/h_canopy_20m` >= 0.5 and `land_segments/canopy/h_canopy_20m` < 200' \
-t0 2023-01-01 -t1 2023-12-31  --strong_beam --cores 20 `

The output files stored at `ICESat_2_puerto_rico` are spatially partitioned - i.e. each file contains ICESat_2 shots with the same parent hexagon at H3 resolution 3.

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

`ih3_read_schema -i ICESat_2_puerto_rico/`

```
|    | column                            | dtype   |
|---:|:----------------------------------|:--------|
|  0 | land_segments/latitude_20m        | float   |
|  1 | orbit_info/sc_orient              | int8    |
|  2 | land_segments/canopy/h_canopy_20m | float   |
|  3 | land_segments/longitude_20m       | float   |
|  4 | land_segments/delta_time          | double  |
|  5 | root_beam                         | string  |
|  6 | geometry                          | binary  |
|  7 | h3_12                             | string  |
```

#
## `ih3_aggregate`

```
usage: ih3_aggregate [-h] -i INPUT -o OUTPUT [-m MAPPER] [-r RES] [-d DROP_COLUMNS [DROP_COLUMNS ...]]
                     [-u USE_COLUMNS [USE_COLUMNS ...]] [-g] [-t] [-f FORMAT] [-n CORES] [-s THREADS] [-A RAM] [-p PORT]

Aggregate icesat-2 shots spatially using h3 (hexagons) or egi (pixels) system

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
  -r RES, --res RES     resolution level of h3/egi system to aggregate to
  -d DROP_COLUMNS [DROP_COLUMNS ...], --drop_columns DROP_COLUMNS [DROP_COLUMNS ...]
                        columns to ignore in aggregation - may be useful for non-dictionary mappers
  -u USE_COLUMNS [USE_COLUMNS ...], --use_columns USE_COLUMNS [USE_COLUMNS ...]
                        only aggregate few columns
  -g, --img             force image output - resamples hexagons if index is h3
  -t, --tiles           export tiles instead of merging outputs to single vector/img file
  -f FORMAT, --format FORMAT
                        output files format [default = parquet]
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 32]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 20]
  -p PORT, --port PORT  port where to open dask dashboard [default = 10000]

```

Aggregate exported shots into vector [.gpkg] or raster [.tif] files at a given resolution. If rasterizing shots indexed with the H3 system, the output pixel size will be equivalent to the hexagons average diameter and their values will be resampled from overlapping hexagon centroids.

### Example:
ih3_aggregate -i /gpfs/data1/vclgp/xiongl/ProjectErrorModel/result/is2_rmse -o /gpfs/data1/vclgp/xiongl/ProjectErrorModel/result/global_is2_error -m mean -r 6 -u rmse 


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
  -y, --quality         apply latest quality filter recipe
  -e, --egi             export shots with EGI spatial index (exact pixels) instead of H3 (approximate hexagons) - much slower!
  -m, --merge           merge outputs and export to single file
  -g, --geo             export file as georreferenced points [.gpkg]
  -l FILLNA, --fillna FILLNA
                        value to replace NAs in the input images
  -d, --dropna          drop NAs before exporting
  -r, --resume          check for files in the output directory and ignore processing for existing files
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 32]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 20]
  -p PORT, --port PORT  port where to open dask dashboard [default = 10000]
```

Extract pixel information at ICESat_2 shot locations.

### Example:

    `ih3_from_img -o temp -i /gpfs/data1/vclgp/xiongl/IS2global/account/HansenData -w 130 -b treecover2000 loss gain lossyear datamask`

