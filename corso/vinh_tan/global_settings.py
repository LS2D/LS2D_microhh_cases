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
from microhhpy.constants import xm_cams

from corso_emissions import Corso_emissions


"""
Case settings.
"""
float_type = np.float64       # KPP does not support float32.

# Mostly for debugging:
sw_openbc = True        # Use open or periodic boundaries.
sw_scalars = True       # Include all scalars used by chemistry.
sw_chemistry = True     # Use KPP chemistry (TODO).
sw_debug = True        # Debug mini domain.


"""
Environment settings.
"""
env_eddy = {
    'era5_path': '/home/scratch1/bart/LS2D_ERA5',
    'cams_path': '/home/scratch1/bart/LS2D_CAMS',
    'lcc_path': '/home/scratch1/bart/LCC/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'microhh_path': '/home/bart/meteo/models/microhh',
    'tuv_path': '/home/bart/meteo/models/microhhpy/external/TUV/V5.4',
    'gpt_path': '/home/bart/meteo/models/coefficients_veerman',
    'cdsapirc': '/home/bart/.cdsapirc',
    'corso_path': '/home/scratch1/bart/emissions/corso',
    'work_path': '.'
}

env_snellius = {
    'era5_path': '/gpfs/work3/0/lesmodels/ls2d_era5',
    'cams_path': '/gpfs/work3/0/lesmodels/ls2d_cams',
    'lcc_path': '/gpfs/work3/0/lesmodels/ls2d_spatial_data/lcc/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'microhh_path': '/home/stratum2/meteo/models/microhh',
    'tuv_path': '/home/stratum2/meteo/models/microhhpy/external/TUV/V5.4',
    'gpt_path': '/home/stratum2/meteo/models/coefficients_veerman',
    'cdsapirc': '/home/stratum2/.cdsapirc',
    'corso_path': '',
    'work_path': '/gpfs/work2/0/nwo21036/bart/CORSO/vinh_tan/'
}

env = env_eddy


"""
LS2D settings.
"""
ls2d_settings = {
    'case_name'   : 'vinh_tan',
    'central_lat' : 11.317,
    'central_lon' : 108.808,
    # This start/end date is only used for the ERA/CAMS download.
    # The start/end date of each domain is set in the `Domain()` definitions below.
    'start_date'  : datetime(year=2021, month=3, day=3, hour=12),
    'end_date'    : datetime(year=2021, month=3, day=6, hour=0),
    'area_size'   : 3,  # Download +/- area_size degree of ERA5/CAMS data.
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

    outer_dom = Domain(
        xsize = 25_600,
        ysize = 25_600,
        itot = 64,
        jtot = 64,
        n_ghost = 3,
        n_sponge = 5,
        lbc_freq = 3600,
        buffer_freq = 3600,
        lon = 108.83,
        lat = 11.34,
        anchor = 'center',
        start_date = datetime(year=2021, month=3, day=3, hour=6),
        end_date = datetime(year=2021, month=3, day=3, hour=8),
        proj_str = proj_str,
        work_dir = f'{env["work_path"]}/outer'
        )

    # Cheating
    outer_dom.npx = 2
    outer_dom.npy = 4

    inner_dom = Domain(
        xsize = 12_800,
        ysize = 12_800,
        itot = 64,
        jtot = 64,
        n_ghost = 3,
        n_sponge = 3,
        lbc_freq = 60,
        buffer_freq = 600,
        parent = outer_dom,
        xstart_in_parent = 1600,
        ystart_in_parent = 1600,
        start_date = datetime(year=2021, month=3, day=3, hour=7),
        end_date = datetime(year=2021, month=3, day=3, hour=8),
        work_dir = f'{env["work_path"]}/inner'
        )

    inner_dom.npx = 2
    inner_dom.npy = 4

    outer_dom.child = inner_dom

    domains = [outer_dom, inner_dom]

else:

    outer_dom = Domain(
        xsize=172_800,
        ysize=172_800,
        itot=576,
        jtot=576,
        n_ghost=3,
        n_sponge=5,
        lbc_freq=3600,
        buffer_freq=3600,
        lon=108.75,
        lat=11.25,
        anchor='center',
        start_date = datetime(year=2021, month=3, day=3, hour=0),
        end_date = datetime(year=2021, month=3, day=3, hour=12),
        proj_str=proj_str,
        work_dir=f'{env["work_path"]}/outer'
        )

    # Cheating
    outer_dom.npx = 32
    outer_dom.npy = 48

    inner_dom = Domain(
        xsize=115_200,
        ysize=115_200,
        itot=1152,
        jtot=1152,
        n_ghost=3,
        n_sponge=3,
        lbc_freq=60,
        buffer_freq=600,
        parent=outer_dom,
        xstart_in_parent=3000,
        ystart_in_parent=3000,
        start_date = datetime(year=2021, month=3, day=3, hour=3),
        end_date = datetime(year=2021, month=3, day=3, hour=12),
        work_dir=f'{env["work_path"]}/inner'
        )

    inner_dom.npx = 64
    inner_dom.npy = 48

    outer_dom.child = inner_dom

    domains = [outer_dom, inner_dom]


"""
Setup vertical grid.
Vertical refinement is supported, by extremely sketchy at the moment. Don't use it :-).
Stretched vertical grids are fine, as long as all domains use the same grid.
"""
vgrid = ls2d.grid.Grid_linear_stretched(kmax=128, dz0=20, alpha=0.007)
zstart_buffer = 0.75 * vgrid.zsize
#vgrid.plot()


"""
Emissions.
Emissions of CO2/NOx/CO are automatically determined from the CORSO emission database.
"""


"""
Variables to download and read from CAMS.
"""
cams_eac4_variables = {
    'eac4_ml': [
         'carbon_monoxide',
         'nitrogen_monoxide',
         'nitrogen_dioxide',
         'nitric_acid',
         'hydrogen_peroxide',
         'ozone',
         'formaldehyde',
         'hydroperoxy_radical',
         'hydroxyl_radical',
         'nitrate_radical',
         'dinitrogen_pentoxide',
         'paraffins',
         'ethene',
         'olefins',
         'isoprene',
         'methanol',
         'ethanol',
         'propane',
         'propene',
         'terpenes',
         'peroxides',
         'methyl_peroxide',
         'methylperoxy_radical',
         'peroxy_acetyl_radical',
         'acetone_product',
         'temperature',
         'specific_humidity'],
    'eac4_sfc': [
        'surface_pressure'],
    }

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

chemical_species = ['co', 'no', 'no2', 'hno3', 'h2o2', 'o3', 'hcho', 'ho2', 'oh', 'no3', 'n2o5', 'rooh', 'c3h6', 'ro2', 'co2']

lumping_species = {
        'c3h6': ['par', 'c2h4', 'ole', 'c5h8', 'ch3oh', 'c2h5oh', 'c3h8', 'c3h6', 'c10h16'],
        'rooh': ['rooh', 'ch3ooh'],
        #'ro2':  ['ch3o2', 'c2o3', 'aco2', 'ic3h7o2', 'hypropo2']}   # Last two species not available in ADS.
        'ro2':  ['ch3o2', 'c2o3', 'aco2']}


if __name__ == '__main__':

    margin = 0.1    # ~10 km

    lon_min = outer_dom.proj.lon.min() + margin
    lon_max = outer_dom.proj.lon.max() - margin

    lat_min = outer_dom.proj.lat.min() + margin
    lat_max = outer_dom.proj.lat.max() - margin

    emiss = Corso_emissions(env['corso_path'])
    emiss.filter_emissions(lon_min, lon_max, lat_min, lat_max)

    # Plot domains and emissions.
    plot_domains(
        [outer_dom, inner_dom],
        scatter_lon=emiss.df_emiss.longitude,
        scatter_lat=emiss.df_emiss.latitude,
        use_projection=True)

    # Check domain decomposition.
    for d in domains:
        check_domain_decomposition(d.itot, d.jtot, vgrid.kmax, d.npx, d.npy)