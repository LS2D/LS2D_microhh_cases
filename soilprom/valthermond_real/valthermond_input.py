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
import os

import matplotlib.pyplot as plt
import numpy as np

import ls2d

from microhhpy import constants
from microhhpy.spatial import Domain
from microhhpy.chem import get_rfmip_species
from microhhpy.io import save_case_input
from microhhpy.io import read_ini, check_ini, save_ini
from microhhpy.land import create_land_surface_input, Land_surface_input

# Import case/system settings.
from settings import env
from settings import ls2d_settings as l2s

# Custom scripts from this directory.
from spatial_tools import is_inside


"""
Case settings.
"""
float_type = np.float32

homogeneous_ls = False       # Realistic or spacially homogeneous land-surface. 
debug_plots = False          # Plot grid, domains, field, BSNEs, ...
work_dir = 'test'

particle_bins = np.array([0, 10, 20, 58, 83, 178]) * 1e-6
particle_list = [f'{particle_bins[i]*1e6:.0f}-{particle_bins[i+1]*1e6:.0f}um' for i in range(particle_bins.size - 1)]
particle_diameter = 0.5 * (particle_bins[1:] + particle_bins[:-1])
    
water_temperature = 273+18  # Not really used in this setup, no significant amounts of open water in domain.
    
# Coordinates of field.
lat_field = np.array([52.875833, 52.875278, 52.870833, 52.871389, 52.875833])
lon_field = np.array([6.932500, 6.931667, 6.937500, 6.938333, 6.932500])

# Coordinates BSNE sampler(s).
lat_bsne = np.array([52.8752778])
lon_bsne = np.array([6.9333333])

    
"""
Define horizontal and vertical grids, and convert coordinates from lat/lon to LES x/y.
"""
lon = l2s['central_lon']
lat = l2s['central_lat']
proj_str = f'+proj=lcc +lat_1={lat-1} +lat_2={lat+1} +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

hgrid = Domain(
    xsize = 51200,
    ysize = 51200,
    itot = 256,
    jtot = 256,
    lon = l2s['central_lon'],
    lat = l2s['central_lat'],
    anchor = 'center',
    proj_str = proj_str)

vgrid = ls2d.grid.Grid_linear_stretched(128, 20, 0.006)

x_column, y_column = hgrid.proj.to_xy(lon_bsne, lat_bsne)
    

"""
Generate ERA5 meteorology with (LS)2D.
"""
# Download files (if needed).
#ls2d.download_era5(l2s)

era = ls2d.Read_era5(l2s)
era.calculate_forcings(n_av=0, method='2nd')

# Interpolate ERA5 onto LES grid.
era5_les = era.get_les_input(vgrid.z)

# Remove top ERA5 level, to ensure that pressure stays
# above the minimum reference pressure in RRTMGP.
era5_les = era5_les.sel(lay=slice(0,135), lev=slice(0,136))

# Time mean.
era5_les_mean = era5_les.mean(dim='time')

# Get background species for RRTMGP from RFMIP
present_day = 0
rrtmgp_species = get_rfmip_species(lat, lon, exp=present_day)


"""
Gravitational settling settings.
Following: https://sci-hub.se/10.1063/1.5022089
NOTE: compared to the paper, this uses a fixed atmospheric density!
"""
rho_p = 2600   # Density particles [kg m-3]
rho_a = 1.225  # Reference density air [kg m-3]
nu = 1.5e-5    # Kinematic viscosity air [m2 s-1]
g = 9.81       # Gravitational acceleration [m s-2]

# New method with mean over bin range:
db = particle_bins
w_s = -rho_p * g / (54 * nu * rho_a) * (db[1:]**2 + db[:-1]*db[1:] + db[:-1]**2)


"""
Create MicroHH `case_input.nc` input.
"""
# Mean radiation profiles on LES grid:
qt_les = era5_les['qt'].mean(axis=0)
o3_les = era5_les['o3'].mean(axis=0)
h2o_les = qt_les / (constants.ep - constants.ep * qt_les)

init_profiles = {
        'z': vgrid.z,
        'thl': era5_les['thl'][0,:],
        'qt': era5_les['qt'][0,:],
        'u': era5_les['u'][0,:],
        'v': era5_les['v'][0,:],
        'nudgefac': np.ones(vgrid.kmax)/10800,
        'o3': o3_les*1e-6,
        'h2o': h2o_les}

radiation  = {
        'z_lay': era5_les['z_lay'  ].mean(axis=0),
        'z_lev': era5_les['z_lev'  ].mean(axis=0),
        'p_lay': era5_les['p_lay'  ].mean(axis=0),
        'p_lev': era5_les['p_lev'  ].mean(axis=0),
        't_lay': era5_les['t_lay'  ].mean(axis=0),
        't_lev': era5_les['t_lev'  ].mean(axis=0),
        'o3':    era5_les['o3_lay' ].mean(axis=0)*1e-6,
        'h2o':   era5_les['h2o_lay'].mean(axis=0)}

# Add background concentrations to init and radiation groups.
for name, conc in rrtmgp_species.items():
    init_profiles[name] = conc
    radiation[name] = conc

timedep_surface = {
        'time_surface': era5_les['time_sec'],
        'p_sbot': era5_les['ps'],
        'thl_sbot': era5_les['wth'],
        'qt_sbot': era5_les['wq']}

timedep_ls = {
        'time_ls': era5_les['time_sec'],
        'u_geo': era5_les['ug'],
        'v_geo': era5_les['vg'],
        'w_ls': era5_les['wls'],
        'thl_ls': era5_les['dtthl_advec'],
        'qt_ls': era5_les['dtqt_advec'],
        'u_ls': era5_les['dtu_advec'],
        'v_ls': era5_les['dtv_advec'],
        'thl_nudge': era5_les['thl'],
        'qt_nudge': era5_les['qt'],
        'u_nudge': era5_les['u'],
        'v_nudge': era5_les['v']}

# NOTE: these soil values are not used by MicroHH when using a realistic land-surface..
soil_index = int(era5_les.type_soil-1)  # -1 = Fortran -> C indexing
soil = {
        'z': era5_les.zs[::-1],
        'theta_soil': era5_les.theta_soil[0,::-1],
        't_soil': era5_les.t_soil[0,::-1],
        'index_soil': np.ones(4) * soil_index,
        'root_frac': era5_les.root_frac_low_veg[::-1]}

save_case_input(
        case_name = 'valthermond',
        init_profiles = init_profiles,
        tdep_surface = timedep_surface,
        tdep_ls = timedep_ls,
        tdep_source = None,
        tdep_chem = None,
        radiation = radiation,
        soil = soil,
        source = None,
        output_dir = work_dir)
    
    
"""
Generate .ini file
"""
ini = read_ini('valthermond.ini.base')

ini['grid']['itot'] = hgrid.itot
ini['grid']['jtot'] = hgrid.jtot
ini['grid']['ktot'] = vgrid.kmax

ini['grid']['xsize'] = hgrid.xsize
ini['grid']['ysize'] = hgrid.ysize
ini['grid']['zsize'] = vgrid.zsize

ini['grid']['lon'] = lon
ini['grid']['lat'] = lat

ini['force']['fc'] = era5_les.fc

ini['buffer']['zstart'] = 0.75 * vgrid.zsize

ini['time']['datetime_utc'] = l2s['start_date'].strftime('%Y-%m-%d %H:%M:%S')
ini['time']['endtime'] = (l2s['end_date']-l2s['start_date']).total_seconds()

ini['cross']['xz'] = hgrid.ysize/2
ini['cross']['yz'] = hgrid.xsize/2
ini['cross']['xy'] = [10,50,100,200,500,1000]

particle_crosses = []
for specie in particle_list:
    particle_crosses.append(specie)
    particle_crosses.append(f'{specie}_path')
ini['cross']['crosslist'] += particle_crosses

ini['column']['coordinates[x]'] = list(x_column)
ini['column']['coordinates[y]'] = list(y_column)

# Switch between spatially homogenous or realistic land-surface:
ini['boundary']['swconstantz0'] = homogeneous_ls
ini['land_surface']['swhomogeneous'] = homogeneous_ls
ini['land_surface']['swwater'] = not homogeneous_ls

# Gravitational settling particles.
ini['fields']['slist'] = particle_list
ini['boundary']['sbot_2d_list'] = particle_list
ini['boundary']['scalar_outflow'] = particle_list
ini['limiter']['limitlist'] = particle_list + ['qt']
ini['advec']['fluxlimit_list'] = particle_list + ['qt']

# Set settling velocity per particle category:
ini['particle_bin']['particle_list'] = particle_list
for i in range(len(particle_list)):
    ini['particle_bin'][f'w_particle[{particle_list[i]}]'] = w_s[i]

# Write new .ini file
check_ini(ini)
save_ini(ini, f'{work_dir}/valthermond.ini')


"""
Create surface dust emissions.
"""
field_mask = np.zeros((hgrid.jtot, hgrid.itot), dtype=bool)

for j in range(hgrid.jtot):
    for i in range(hgrid.itot):
        if is_inside(hgrid.proj.lon[j,i], hgrid.proj.lat[j,i], lon_field, lat_field, lon_field.size):
            field_mask[j,i] = True

field_flux = np.zeros((hgrid.jtot, hgrid.itot), dtype=float_type)
field_flux[field_mask] = 1.

# All scalars get the same emission.
for scalar in particle_list:
    field_flux.tofile(f'{work_dir}/{scalar}_bot_in.0000000')



if not homogeneous_ls:
    """
    Create realistic land-use input from 100 m CORINE.
    """

    # Default soil depths IFS.
    z_soil = np.array([-0.035, -0.175, -0.64 , -1.945])[::-1]

    lu_lcc = create_land_surface_input(
        hgrid.proj.lon,
        hgrid.proj.lat,
        z_soil,
        land_use_source='corine_100m',
        land_use_tiff=env['corine_tif'],
        save_binaries=True,
        output_dir=work_dir,
        save_netcdf=True,
        netcdf_file='lsm_input.nc')


    # TODO: Init soil from HiHydroSoil. For now spatially homogeneous.
    soil = Land_surface_input(
        hgrid.itot,
        hgrid.jtot,
        4,
        exclude_veg=True,
        debug=True,
        float_type=float_type
    )

    soil.theta_soil[:,:,:] = era5_les_mean.theta_soil.values[::-1, None, None]
    soil.t_soil[:,:,:] = era5_les_mean.t_soil.values[::-1, None, None]
    soil.index_soil[:,:,:] = int(era5_les_mean.type_soil) - 1  # FORTRAN -> C
    soil.to_binaries(path=work_dir, allow_overwrite=True)



"""
Link HTESSEL and RTE+RRTMGP lookup tables from MicroHH repo.
"""
microhh_path = env['microhh_path']
rrtmgp_path = f'{microhh_path}/rte-rrtmgp-cpp/'

lookup_tables = [
    (f'{microhh_path}/misc/van_genuchten_parameters.nc', 'van_genuchten_parameters.nc'),
    (f'{rrtmgp_path}/rrtmgp-data/rrtmgp-gas-lw-g128.nc', 'coefficients_lw.nc'),
    (f'{rrtmgp_path}/rrtmgp-data/rrtmgp-gas-sw-g112.nc', 'coefficients_sw.nc'),
    (f'{rrtmgp_path}/rrtmgp-data/rrtmgp-clouds-lw.nc', 'cloud_coefficients_lw.nc'),
    (f'{rrtmgp_path}/rrtmgp-data/rrtmgp-clouds-sw.nc', 'cloud_coefficients_sw.nc')]

for src, name in lookup_tables:
    dst = f'{work_dir}/{name}'
    if not os.path.exists(dst):
      os.symlink(src, dst)


if debug_plots:

    # Vertical grid:
    vgrid.plot()

    # Horizontal domain, field, BSNE, ...
    plt.figure()
    plt.plot(hgrid.proj.bbox_lon, hgrid.proj.bbox_lat, 'r--', label='domain')
    plt.plot(lon_field, lat_field, 'k-', label='field')
    plt.scatter(lon_bsne, lat_bsne, label='output columns')
    plt.legend()