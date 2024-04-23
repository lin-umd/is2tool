# icesat-2 atl08 h3 command tool
### *Author: Lin xiong, Tiago de Conto*
### *April 23, 2024*

<br>

The GEDI/H3 database is a spatially indexed version of the GEDI database (L2A, L2B and L4A products only). It was build to make querying and processing GEDI shots more efficient, especially when targeting specific locations. The following CLI tools make the GEDI/H3 database easily accessible and provide some basic post processing capabilities useful to most users of the UMD/GEOG cluster:

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
conda activate /gpfs/data1/vclgp/decontot/environments/gedih3
```
#
## `ih3_list_variables`

```
usage: ih3_list_variables [-h] [-a] [-g GREP] [-p PRODUCTS [PRODUCTS ...]]

List variables available in the GEDI H3 database

options:
  -h, --help            show this help message and exit
  -a, --ancillary       include variables from ancillary database
  -g GREP, --grep GREP  match substring
  -p PRODUCTS [PRODUCTS ...], --products PRODUCTS [PRODUCTS ...]
                        search only specified GEDI [l2a, l2b, l4a] and/or ancillary products
```

List variables available in the GEDI/H3 database as a 3 column table:
- *product*: product containing the varaible
- *column*: column corresponding to the variable in the GEDI/H3 product data frame
- *dtype*: data type

`ancillary` data were extracted from sources outside the GEDI and matched to each GEDI shot. Ancillaru data can be added upon request. Currently the following  products are available:
- `quality_flags`: flags used for filtering quality GEDI shots following the most up-to-date recipe
- `wsci_v1`: waveform structural complexity index version 1
- `wsci_v1_usa`: waveform structural complexity index version 1 from US exclusive models
- [`glad_forest_loss`](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2021_v1_9): forest loss information from the Landsat derived product by Hansen et al (2012) at 30m resolution
- [`esa_land_cover`](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100): land cover classification from ESA/sentinel at 10m resolution
- [`nasa_dem`](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001): topography information from the NASA dem product at 30m resolution

### Example:

`ih3_list_variables -a -g shot_number`

```
| product          | column      | dtype   |
|:-----------------|:------------|:--------|
| l2a              | shot_number | uint64  |
| l2b              | shot_number | uint64  |
| l4a              | shot_number | uint64  |
| quality_flags    | shot_number | uint64  |
| wsci_v1_usa      | shot_number | uint64  |
| glad_forest_loss | shot_number | uint64  |
| wsci_v1          | shot_number | uint64  |
```

#
## `ih3_list_resolutions`

```
usage: ih3_list_resolutions [-h] [-3] [-e]

List resolution specs for H3 and EGI spatial indexing systems

options:
  -h, --help  show this help message and exit
  -3, --h3    list H3 system only
  -e, --egi   list EGI system only
```

List resolutions available for aggregation of GEDI products using H3 and EGI spatial indexing systems.

H3 is the [hexagon hierarchical spatial index](https://h3geo.org/) developed by Uber. EGI stands for Ease Grid Index - a custom spatial index system with perfectly nested squares based on the EASE-Grid 2.0 Global CRS (EPSG:6933) - developed by yours truly ;)

Extracting EGI indexed shots is slower than using H3. However, it is useful when precise raster outputs are desired - i.e. pixel values with exact GEDI shot intersections for any given resolution and no resampling. The EGI was also designed to match the image specifications of the GEDI L4B product at 1km scale - thus raster outputs using the EGI system follow GEDI product release standards.

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

## -- ease grid index (EGI) levels - square pixels
       pixel size (m)
level                
1               1.001
2               5.005
3              25.022
4             100.090
5             200.179
6            1000.895
7            2001.790
8           10008.950
9           20017.900
10          40035.801
11          80071.602
12         160143.204
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

Quick access to 3 letter country codes. Those countries boundarias are available in the cluster's postgres database and may be used to query GEDI shots with `ih3_extract_shots`.

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
usage: ih3_extract_shots [-h] -o OUTPUT [-r REGION] [-l2a L2A [L2A ...]] [-l2b L2B [L2B ...]] [-l4a L4A [L4A ...]] [-a ANCI] [-q QUERY] [-y] [-e] [-m] [-g]
                         [-n CORES] [-s THREADS] [-A RAM] [-p PORT]

Filter and export spatially indexed GEDI shots from multiple products using the h3 (hexagons) or egi (pixels) system

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output directory or file path
  -r REGION, --region REGION
                        path to vector (.shp, .gpkg, .kml etc.) or raster (.tif) file with region of interest to extract shots from OR iso3 country code -
                        if not provided, all land surface data will be queried
  -l2a L2A [L2A ...], --l2a L2A [L2A ...]
                        GEDI L2A variables to export
  -l2b L2B [L2B ...], --l2b L2B [L2B ...]
                        GEDI L2B variables to export
  -l4a L4A [L4A ...], --l4a L4A [L4A ...]
                        GEDI L4A variables to export
  -a ANCI, --anci ANCI  quoted dictionary of ancillary variables to export - e.g. "{'glad_forest_loss':['loss','lossyear']}"
  -q QUERY, --query QUERY
                        single string with custom filters upon listed variables - use pandas.DataFrame.query notation
  -y, --quality         apply latest quality filter recipe
  -e, --egi             export shots with EGI spatial index (exact pixels) instead of H3 (approximate hexagons) - much slower!
  -m, --merge           merge outputs and export to single file
  -g, --geo             export file as georreferenced points [.gpkg]
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 20]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -A RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 10]
  -p PORT, --port PORT  port where to open dask dashboard [default = 10000]
```

Extract spatially indexed GEDI shots for a given region (or globally), filtered or not, geolocated or not, using distributed processes with RAM thresholding. 

If later aggregation of shots in large regions will be performed with `ih3_aggregate`, avoid passing the `-m, --merge` flag, since merged outputs are not processed in parallel.

You can follow the progress of your queries through the dask dashboard at the `-p, --port` adress. For example, in order to access it locally at the CLI's default port 10000 grom gsapp8, connect to the your desired GEOG cluster forwarding the desired port as in:

```
ssh -L 10000:localhost:10000 <username>@gsapp8
```

Then, once the `ih3` tool is running, navigate to `localhost:10000` in the browser of your local machine.

### Example:

Let's query foliage height diversity, canopy cover, biomass and structural complexity from high quality geolocated GEDI shots over Mexico and export the results to a directory called `gedi_mexico`: 

`ih3_extract_shots -o gedi_mexico -r MEX -l2b fhd_normal cover_z_000 -l4a agbd -a "{'wsci_v1':['wsci_pft']}" --quality --geo`

```
## -- dask dashboard available at: http://127.0.0.1:10000/status
## -- loading region of interest
## -- reading distributed data frame
## -- scheduling georreferencing
## -- loading and exporting data
[########################################] | 100% Completed |  7min 47.5s
## -- DONE
```

The output files stored at `gedi_mexico` are spatially partitioned - i.e. each file contains GEDI shots with the same parent hexagon at H3 resolution 3.

#
## `ih3_read_schema`

```
usage: ih3_read_schema [-h] -i INPUT

List variables available in the GEDI H3 database

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        directory or file path [parquet or gpkg] - generated by `ih3_extract_shots`
```

Print the column names and data types of exported GEDI/H3 datasets.

### Example:

`ih3_read_schema -i gedi_mexico/`

```
|    | column         | dtype   |
|---:|:---------------|:--------|
|  0 | h3_12          | str     |
|  1 | fhd_normal     | float   |
|  2 | cover_z_000    | float   |
|  3 | lon_lowestmode | float   |
|  4 | lat_lowestmode | float   |
|  5 | shot_number    | int     |
|  6 | agbd           | float   |
|  7 | wsci_pft       | float   |
|  8 | quality_flag   | int     |
```

#
## `ih3_aggregate`

```
usage: ih3_aggregate [-h] -i INPUT -o OUTPUT [-m MAPPER] [-r RES] [-d DROP_COLUMNS [DROP_COLUMNS ...]] [-u USE_COLUMNS [USE_COLUMNS ...]] [-g] [-t] [-n CORES]
                     [-s THREADS] [-a RAM] [-p PORT]

Aggregate GEDI shots spatially using h3 (hexagons) or egi (pixels) system

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input file or directory with indexed GEDI shots
  -o OUTPUT, --output OUTPUT
                        output directory or file name
  -m MAPPER, --mapper MAPPER
                        how to aggregate shots under pandas standard - string with single function name or quoted list/dictionary: 'mean' or "['mean', 'std',
                        'count']" or "{'agbd':['mean','count'], 'sensitivity':['mean']}"
  -r RES, --res RES     resolution level of h3/egi system to aggregate to
  -d DROP_COLUMNS [DROP_COLUMNS ...], --drop_columns DROP_COLUMNS [DROP_COLUMNS ...]
                        columns to ignore in aggregation - may be useful for non-dictionary mappers
  -u USE_COLUMNS [USE_COLUMNS ...], --use_columns USE_COLUMNS [USE_COLUMNS ...]
                        only aggregate few columns
  -g, --img             force image output - resamples hexagons if index is h3
  -t, --tiles           export tiles instead of merging outputs to single vector/img file
  -n CORES, --cores CORES
                        number of cpu cores to use [default = 20]
  -s THREADS, --threads THREADS
                        number of threads per cpu [default = 1]
  -a RAM, --ram RAM     maximum RAM usage per cpu - in Giga Bytes [default = 10]
  -p PORT, --port PORT  port where to open dask dashboard [default = 10000]
```

Aggregate exported shots into vector [.gpkg] or raster [.tif] files at a given resolution. If rasterizing shots indexed with the H3 system, the output pixel size will be equivalent to the hexagons average diameter and their values will be resampled from overlapping hexagon centroids.

### Example:

Let's aggregate the mexico shots:

1. vector file (mex_hex.gpkg) with mean values at resolution 7 (~3km diameter hexes):
    
    `ih3_aggregate -i gedi_mexico/ -o mex_hex -m mean -r 7 -u agbd fhd_normal cover_z_000 wsci_pft`

2. raster file (mex_agbd.tif) of agbd mean and standard deviation at resolution 8 (~1km pixels):

    `ih3_aggregate -i gedi_mexico/ -o mex_agbd -m "['mean','std']" -r 8 -u agbd -g`


#
## `ih3_from_img`

```
usage: ih3_from_img [-h] -o OUTPUT -i IMG [-f FORMAT] [-b BAND_NAMES [BAND_NAMES ...]] [-w WINDOW_OPERATIONS [WINDOW_OPERATIONS ...]] [-y] [-e] [-m] [-g] [-l FILLNA] [-d] [-r] [-n CORES] [-s THREADS] [-A RAM] [-p PORT]

Incorporate information from image data to GEDI shots

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
                        (optional) moving window operations to apply to image bands: list of 3 integer numbers representing band number (0 indexed), window size (1-9) and operation id (0 = sum, 1 = mean, 2 = median, 3 = mode), respectively; e.g. -w 033 151 152
  -y, --quality         apply latest quality filter recipe
  -e, --egi             export shots with EGI spatial index (exact pixels) instead of H3 (approximate hexagons) - much slower!
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
  -p PORT, --port PORT  port where to open dask dashboard [default = 10000]
```

Extract pixel information at GEDI shot locations.

### Example:

Let's extract pixel values from a nasa DEM tile with ~30m resolution. 

1. First, check out the band information using gdal:

    `gdalinfo /gpfs/data1/vclgp/decontot/temp/ih3_tutorial/nasa_dem_tile.tif`

    ```
    Driver: GTiff/GeoTIFF
    Files: /gpfs/data1/vclgp/decontot/temp/ih3_tutorial/nasa_dem_tile.tif
    Size is 20480, 20480
    Coordinate System is:
    GEOGCRS["WGS 84",
        ENSEMBLE["World Geodetic System 1984 ensemble",
            MEMBER["World Geodetic System 1984 (Transit)"],
            MEMBER["World Geodetic System 1984 (G730)"],
            MEMBER["World Geodetic System 1984 (G873)"],
            MEMBER["World Geodetic System 1984 (G1150)"],
            MEMBER["World Geodetic System 1984 (G1674)"],
            MEMBER["World Geodetic System 1984 (G1762)"],
            MEMBER["World Geodetic System 1984 (G2139)"],
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]],
            ENSEMBLEACCURACY[2.0]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        CS[ellipsoidal,2],
            AXIS["geodetic latitude (Lat)",north,
                ORDER[1],
                ANGLEUNIT["degree",0.0174532925199433]],
            AXIS["geodetic longitude (Lon)",east,
                ORDER[2],
                ANGLEUNIT["degree",0.0174532925199433]],
        USAGE[
            SCOPE["Horizontal component of 3D system."],
            AREA["World."],
            BBOX[-90,-180,90,180]],
        ID["EPSG",4326]]
    Data axis to CRS axis mapping: 2,1
    Origin = (80.403978999703043,38.229423883217663)
    Pixel Size = (0.000269494585236,-0.000269494585236)
    Metadata:
      AREA_OR_POINT=Area
    Image Structure Metadata:
      COMPRESSION=LZW
      INTERLEAVE=PIXEL
    Corner Coordinates:
    Upper Left  (  80.4039790,  38.2294239) ( 80d24'14.32"E, 38d13'45.93"N)
    Lower Left  (  80.4039790,  32.7101748) ( 80d24'14.32"E, 32d42'36.63"N)
    Upper Right (  85.9232281,  38.2294239) ( 85d55'23.62"E, 38d13'45.93"N)
    Lower Right (  85.9232281,  32.7101748) ( 85d55'23.62"E, 32d42'36.63"N)
    Center      (  83.1636036,  35.4697993) ( 83d 9'48.97"E, 35d28'11.28"N)
    Band 1 Block=256x256 Type=Float32, ColorInterp=Gray
      Description = elevation
    Band 2 Block=256x256 Type=Float32, ColorInterp=Undefined
      Description = slope
    Band 3 Block=256x256 Type=Float32, ColorInterp=Undefined
      Description = aspect
    Band 4 Block=256x256 Type=Float32, ColorInterp=Undefined
      Description = water_mask
    ```

2. Extract pixel values from all bands + average elevation and slope on 3x3 windows around each quality shot and export as georreferenced vector files:
    
    `ih3_from_img -o shots_from_img -i /gpfs/data1/vclgp/decontot/temp/ih3_tutorial/nasa_dem_tile.tif -w 031 131 -y -g -d`

2. Check the results:

    `ih3_read_schema -i shots_from_img/`

    ```
    |    | column                  | dtype   |
    |---:|:------------------------|:--------|
    |  0 | h3_12                   | str     |
    |  1 | shot_number             | int     |
    |  2 | relative_pixel_distance | float   |
    |  3 | elevation               | float   |
    |  4 | slope                   | float   |
    |  5 | aspect                  | float   |
    |  6 | water_mask              | float   |
    |  7 | elevation_mean_3x3      | float   |
    |  8 | slope_mean_3x3          | float   |
    ```