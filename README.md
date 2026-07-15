# (LS)²D MicroHH cases
Public repository of MicroHH cases using (LS)²D

## Datasets
- **Land-use:** [CORINE land cover](https://land.copernicus.eu/en/products/corine-land-cover/clc2018#download) (raster) ([mirror](https://drive.proton.me/urls/HV1TJ5X3EM#JdzBtsuQwUdg))
- **Land-use:** [Copernicus Global Land Cover](https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif?download=1)

## Lookup tables
The following lookup tables need to be copied or symlinked into the run directory (source → link name):

Van Genuchten soil hydraulic parameters for the land-surface scheme:
- `microhh/misc/van_genuchten_parameters.nc` → `van_genuchten_parameters.nc`

RRTMGP longwave gas optics coefficients:
- `microhh/rte-rrtmgp-cpp/rrtmgp-data/rrtmgp-gas-lw-g128.nc` → `coefficients_lw.nc`
- `microhh/rte-rrtmgp-cpp/rrtmgp-data/rrtmgp-gas-sw-g112.nc` → `coefficients_sw.nc`

RRTMGP longwave cloud optical properties:
- `microhh/rte-rrtmgp-cpp/rrtmgp-data/rrtmgp-clouds-lw.nc` → `cloud_coefficients_lw.nc`
- `microhh/rte-rrtmgp-cpp/rrtmgp-data/rrtmgp-clouds-sw.nc` → `cloud_coefficients_sw.nc`
