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
float_type = np.float64
sw_debug = False               # Small debug/test domain.

# Location MMIJ tower.
lat_mmij = 52.848167
lon_mmij = 3.435667

# Depression passing over MMIJ:
start_date = datetime(year=2012, month=9, day=23, hour=12)
end_date   = datetime(year=2012, month=9, day=25, hour=12)


"""
Environment settings.
"""
env_eddy = {
    'era5_path': '/home/scratch1/bart/LS2D_ERA5',
    'microhh_path': '/home/bart/meteo/models/microhh',
    'gpt_path': '/home/bart/meteo/models/coefficients_veerman',
    'cdsapirc': '/home/bart/.cdsapirc',
    'work_path': 'test'
}

env_snellius = {
    'era5_path': '/gpfs/work3/0/lesmodels/team_bart/ls2d_era5',
    'microhh_path': '/home/bstratum/meteo/models/microhh',
    'gpt_path': '/gpfs/work3/0/lesmodels/team_bart/coefficients_veerman',
    'cdsapirc': '/home/bstratum/.cdsapirc',
    'work_path': '/scratch-shared/bstratum/corso/tata_steel'
}

hpcperm = '/hpcperm/nkbs'
scratch = '/scratch/nkbs'
home = '/home/nkbs'
env_ecmwf = {
    'era5_path': f'{hpcperm}/LS2D_ERA5/',
    'microhh_path': f'{home}/meteo/models/microhh',
    'gpt_path': f'{home}/meteo/models/coefficients_veerman',
    'cdsapirc': f'{home}/.cdsapirc',
    'work_path': f'{scratch}/mmij_v1/',
}

env = env_ecmwf


"""
LS2D settings.
"""
ls2d_settings = {
    'case_name'   : 'mmij',
    'central_lat' : lat_mmij,
    'central_lon' : lon_mmij,
    'start_date'  : start_date,
    'end_date'    : end_date,
    'area_size'   : 4,  # Download +/- area_size degree of ERA5/CAMS data.
    'era5_path'   : env['era5_path'],
    'cdsapirc'    : env['cdsapirc'],
    'write_log'   : False,
    'data_source' : 'CDS',
    'era5_expver' : 1,  # 1=normal, 5=near-realtime ERA5.
    }


"""
Stretched vertical grid.
"""
vgrid = ls2d.grid.Grid_linear_stretched(kmax=128, dz0=25, alpha=0.02)
zstart_buffer = 0.75 * vgrid.zsize


"""
Setup domains with projection and nesting.
"""
lon = ls2d_settings['central_lon']
lat = ls2d_settings['central_lat']
proj_str = f'+proj=lcc +lat_1={lat-1} +lat_2={lat+1} +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

if sw_debug:
    xsize = 32_000
    ysize = 32_000

    itot = 64
    jtot = 64

    npx = 2
    npy = 4

else:
    xsize = 256*400
    ysize = 256*400

    itot = 256
    jtot = 256

    npx = 16
    npy = 16

outer_dom = Domain(
    xsize = xsize,
    ysize = ysize,
    itot = itot,
    jtot = jtot,
    n_ghost = 3,
    n_sponge = 5,
    lbc_freq = 3600,
    buffer_freq = 3600,
    lat = lat_mmij,
    lon = lon_mmij,
    anchor = 'center',
    start_date = start_date,
    end_date = end_date,
    proj_str = proj_str,
    work_dir = env['work_path']
    )

# Cheating
outer_dom.npx = npx
outer_dom.npy = npy

domains = [outer_dom]



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Plot domains and emissions.
    fig, ax = plot_domains(
        domains,
        use_projection=True,
        osm_background=True,
        zoom_level=9)

    # Plot vertical grid.
    vgrid.plot()

    # Check domain decomposition.
    for d in domains:
        check_domain_decomposition(d.itot, d.jtot, vgrid.kmax, d.npx, d.npy)
