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

from datetime import datetime

"""
Switch between different systems:
"""

project = '/gpfs/work3/0/lesmodels/'
env_snellius = {
        'system': 'snellius',
        'era5_path': f'{project}/ls2d_era5',
        'microhh_path': None,
        'corine_tif': f'{project}/ls2d_spatial_data/corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
        }

env_eddy = {
        'system': 'eddy',
        'era5_path': '/home/scratch1/bart/LS2D_ERA5/',
        'microhh_path': '/home/bart/meteo/models/microhh',
        'corine_tif': '/home/scratch1/bart/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
        }

env_stormy = {
        'system': 'stormy',
        'era5_path': '/home/scratch1/meteo_data/LS2D_ERA5/',
        'microhh_path': '/home/bart/meteo/models/microhh',
        'corine_tif': '/home/scratch1/meteo_data/Corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif',
        }

# Switch between environments.
env = env_stormy

# Dictionary with settings
ls2d_settings = {
    'central_lon' : 6.932500,
    'central_lat' : 52.875833,
    'start_date'  : datetime(year=2022, month=5, day=10, hour=8),
    'end_date'    : datetime(year=2022, month=5, day=10, hour=14),
    'area_size'   : 1.5,
    'case_name'   : 'valthermond',
    'era5_path'   : env['era5_path'],
    'era5_expver' : 1,
    'write_log'   : False,
    'data_source' : 'CDS',
    }
