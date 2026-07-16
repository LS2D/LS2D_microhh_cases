#
# This file is part of LS2D.
#
# Copyright (c) 2017-2025 Wageningen University & Research
# Author: Bart van Stratum (WUR)
#
# LS2D is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LS2D is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LS2D.  If not, see <http://www.gnu.org/licenses/>.
#

import sys

from datetime import datetime
import numpy as np

# pip install ls2d
import ls2d

# pip install microhhpy
from microhhpy.spatial import Domain, plot_domains
from microhhpy.utils import check_domain_decomposition


"""
Case settings.
"""
float_type = np.float32       # KPP does not support float32.
sw_debug = True      # Debug mini domain.


"""
Environment settings.
"""
env_eddy = {
    'era5_path': '/home/scratch1/bart/LS2D_ERA5',
    'cams_path': '/home/scratch1/bart/LS2D_CAMS',
    'lcc_path': '/home/scratch1/bart/LCC/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'corine_path': '/home/scratch1/bart/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
    'microhh_path': '/home/bart/meteo/models/microhh',
    'gpt_path': '/home/bart/meteo/models/coefficients_veerman',
    'cdsapirc': '/home/bart/.cdsapirc',
    'work_path': 'test'
}

env_stormy = {
    'era5_path': '/home/scratch1/meteo_data/LS2D_ERA5',
    'cams_path': '/home/scratch1/meteo_data/LS2D_CAMS',
    'lcc_path': '/home/scratch1/meteo_data/LCC/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'corine_path': '/home/scratch1/meteo_data/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
    'microhh_path': '/home/bart/meteo/models/microhh',
    'gpt_path': '/home/bart/meteo/models/coefficients_veerman',
    'cdsapirc': '/home/bart/.cdsapirc',
    'work_path': 'test'
}

env_snellius = {
    'era5_path': '/gpfs/work3/0/lesmodels/team_bart/ls2d_era5',
    'cams_path': '/gpfs/work3/0/lesmodels/team_bart/ls2d_cams',
    'lcc_path': '/gpfs/work3/0/lesmodels/team_bart/ls2d_spatial_data/lcc/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'microhh_path': '/home/bstratum/meteo/models/microhh',
    'gpt_path': '/gpfs/work3/0/lesmodels/team_bart/coefficients_veerman',
    'cdsapirc': '/home/bstratum/.cdsapirc',
    'work_path': '/scratch-shared/bstratum/corso/tata_steel'
}

env = env_stormy


"""
LS2D settings.
"""
ls2d_settings = {
    'case_name'   : 'rheinisches-revier',
    'central_lat' : 50.9465,
    'central_lon' : 6.5801,
    # This start/end date is only used for the ERA/CAMS download.
    # The start/end date of each domain is set in the `Domain()` definitions below.
    'start_date'  : datetime(year=2007, month=8, day=4, hour=0),
    'end_date'    : datetime(year=2007, month=8, day=4, hour=23),
    'area_size'   : 1,  # Download +/- area_size degree of ERA5/CAMS data.
    'era5_path'   : env['era5_path'],
    'cams_path'   : env['cams_path'],
    'cdsapirc'    : env['cdsapirc'],
    'write_log'   : False,
    'data_source' : 'CDS',
    'era5_expver' : 1,  # 1=normal, 5=near-realtime ERA5.
    }


"""
Setup domains with projection and nesting.
"""
lon = ls2d_settings['central_lon']
lat = ls2d_settings['central_lat']
proj_str = f'+proj=lcc +lat_1={lat-1} +lat_2={lat+1} +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

if sw_debug:
    print('Using debug domain...')

    domain = Domain(
        xsize = 3_200,
        ysize = 3_200,
        itot = 32,
        jtot = 32,
        #lon = 6.62,
        #lat = 51.02,
        lon = 6.6685,   # Niederaussem only
        lat = 50.9946,
        anchor = 'center',
        start_date = datetime(year=2007, month=8, day=4, hour=8),
        end_date = datetime(year=2007, month=8, day=4, hour=15),
        proj_str = proj_str,
        work_dir = env['work_path']
        )

    # Cheating
    domain.npx = 1
    domain.npy = 1

    # Vertical grid.
    vgrid = ls2d.grid.Grid_linear_stretched(kmax=96, dz0=20, alpha=0.01)
    zstart_buffer = 0.75 * vgrid.zsize
    #vgrid.plot()

#else:
#    print('Using production domain...')
#
#    domain = Domain(
#        xsize=115_200,
#        ysize=115_200,
#        itot=1152,
#        jtot=1152,
#        lon=86.45,
#        lat=22.55,
#        anchor='center',
#        start_date = datetime(year=2021, month=2, day=23, hour=12),
#        end_date = datetime(year=2021, month=2, day=25, hour=12),
#        proj_str=proj_str,
#        work_dir=env['work_path']
#        )
#
#    # Cheating
#    domain.npx = 24
#    domain.npy = 32
#
#    domains = [domain]
#
#    # Vertical grid.
#    vgrid = ls2d.grid.Grid_linear_stretched(kmax=96, dz0=20, alpha=0.015)
#    zstart_buffer = 0.75 * vgrid.zsize


"""
Variables to download and read from CAMS.
"""
# NOTE to self: only available <2021.
cams_egg4_variables = {
    'egg4_ml': [
        'carbon_dioxide',
        'methane',
        'temperature',
        'specific_humidity'],
    'egg4_sl': [
        'logarithm_of_surface_pressure']
    }


chemical_species = ['co2']


"""
Emission settings.
NA = Niederaussem
...
"""
stacks = {
    'NA_D': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99353, 'lon': 6.66737},
    'NA_E': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99451, 'lon': 6.66645},
    'NA_F': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99400, 'lon': 6.66878},
    'NA_G': {'fuel': 'lignite_cooling_tower', 'p':  600, 'eta': 0.31, 'rh': 0.95, 'h': 128, 'lat': 50.99514, 'lon': 6.66810},
    'NA_H': {'fuel': 'lignite_cooling_tower', 'p':  600, 'eta': 0.31, 'rh': 0.95, 'h': 128, 'lat': 50.99465, 'lon': 6.67041},
    'NA_K': {'fuel': 'lignite_cooling_tower', 'p': 1027, 'eta': 0.43, 'rh': 0.95, 'h': 200, 'lat': 50.99611, 'lon': 6.67137},
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Approximate coordinates of power plants, just for plotting...
    sites = {
            'Niederaußem':  (50.9891, 6.6679),
            'Neurath':      (51.0346, 6.6153),
            'Weisweiler':   (50.8360, 6.3175),
            'Frimmersdorf': (51.0570, 6.5765),
            'Knapsack':     (50.8620, 6.8428),
            }

    lat, lon = np.array(list(sites.values())).T

    # Plot horizontal domain with emissions.
    margin = 0.01    # ~10 km

    lon_min = domain.proj.lon.min() + margin
    lon_max = domain.proj.lon.max() - margin

    lat_min = domain.proj.lat.min() + margin
    lat_max = domain.proj.lat.max() - margin

    # Plot domains and emissions.
    fig, ax = plot_domains(
        domains,
        use_projection=True,
        background='osm',
        zoom_level=16,
        labels=['Outer domain'])

    for name, p in stacks.items():
        ax.scatter(p['lon'], p['lat'], label=f'{name}: P={p["p"]} MW, z={p["h"]} m', transform=ccrs.PlateCarree())

    plt.legend(loc='upper left')

    # Plot vertical grid.
    #vgrid.plot()

    # Check domain decomposition.
    for d in domains:
        check_domain_decomposition(d.itot, d.jtot, vgrid.kmax, d.npx, d.npy)
