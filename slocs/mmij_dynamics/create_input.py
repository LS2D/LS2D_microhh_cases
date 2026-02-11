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

from datetime import timedelta
import argparse
import shutil
import sys
import os
import logging

import pandas as pd
import numpy as np

# pip install ls2d
import ls2d

# pip install microhhpy
from microhhpy.real import create_input_from_regular_latlon
from microhhpy.real import create_3d_geowind_from_regular_latlon
from microhhpy.real import create_sst_from_regular_latlon
from microhhpy.real import create_2d_coriolis_freq
from microhhpy.real import regrid_les
from microhhpy.real import link_bcs_from_parent, link_buffer_from_parent

from microhhpy.thermo import calc_moist_basestate, read_moist_basestate, save_moist_basestate
from microhhpy.thermo import save_basestate_density, read_basestate_density
from microhhpy.thermo import qsat, exner

from microhhpy.io import read_ini, check_ini, save_ini, save_case_input
from microhhpy.chem import get_rfmip_species
from microhhpy.spatial import calc_vertical_grid_2nd
from microhhpy.constants import xm_cams
from microhhpy.logger import logger

# Local settings and scripts.
import global_settings as settings

logger = logging.getLogger("microhhpy")
logger.setLevel(logging.INFO)   # INFO/DEBUG


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--domain',
        choices=['inner', 'outer'],
        required=True)
    args = parser.parse_args()

    return args


def read_era5_cams(ls2d_settings, start_date, end_date, vgrid):
    """
    Read / process ERA5 and CAMS data using (LS)2D.
    Returns the 3D fields (needed for open boundaries),
    and vertical profiles (needed for e.g. basestate).
    """
    logger.info('Reading ERA5 and CAMS using (LS)2D')

    ls2d_settings['start_date'] = start_date
    ls2d_settings['end_date'] = end_date

    era5 = ls2d.Read_era5(ls2d_settings)
    era5.calculate_forcings(n_av=2, method='2nd')
    era5_les = era5.get_les_input(vgrid.z)

    # Remove top ERA5 level, to stay within minium reference pressure RRTMGP.
    era5_les = era5_les.sel(lay=slice(0,135), lev=slice(0,136))

    # Mean profiles, only used for base-state density and dummy input model.
    era5_mean = era5_les.mean(dim='time')

    return era5, era5_les, era5_mean


def create_basestate(era5_mean, vgrid):
    """
    Create base-state density LES. With open-boundaries it is important that this
    density matches the one used by the model, otherwise the momentum fields
    won't be divergence free!
    """
    logger.info('Creating base state density')

    bs = calc_moist_basestate(
        era5_mean.thl.values,
        era5_mean.qt.values,
        era5_mean.ps.values,
        vgrid.z,
        vgrid.zsize,
        settings.float_type)

    return bs


def create_init_and_bcs_outer(era5, domain, bs):
    """
    Create the initial 3D fields interpolated from ERA5,
    together with the lateral/top boundary conditions.
    """
    logger.info('Creating initial and boundary conditions')

    # Gaussian filter size (m) of filter after interpolation.
    sigma_h = 10_000

    time_sec = era5.time_sec

    fields_era = {
        'u': era5.u[:,:,:,:],
        'v': era5.v[:,:,:,:],
        'w': era5.wls[:,:,:,:],
        'thl': era5.thl[:,:,:,:],
        'qt': era5.qt[:,:,:,:],
    }

    create_input_from_regular_latlon(
        fields_era,
        era5.lons.data,   # Strip off array masks.
        era5.lats.data,
        era5.z[:,:,:,:],
        era5.p[:,:,:,:],
        time_sec,
        settings.vgrid.z,
        settings.vgrid.zsize,
        settings.zstart_buffer,
        bs['rho'],
        bs['rhoh'],
        domain,
        sigma_h,
        perturb_size=4,
        perturb_amplitude={'thl': 0.1, 'qt':0.1e-3},
        perturb_max_height=1000,
        clip_at_zero=['qt'],
        name_suffix='ext',
        output_dir=domain.work_dir,
        ntasks=8,
        save_netcdf=True,
        float_type=settings.float_type)



def create_init_and_bcs_inner(domain, bs, vgrid):

    gd = calc_vertical_grid_2nd(vgrid.z, vgrid.zsize, float_type=float)

    # Create and/or link initial fields and boundary conditions from parent domain.
    start_time = int((domain.start_date - domain.parent.start_date).total_seconds())
    end_time = int(start_time + (domain.end_date - domain.start_date).total_seconds())
    time_offset = -start_time

    # Initial 3D fields at (optionally) delayed start time of child domain.
    fields = ['u', 'v', 'w', 'thl', 'qt']
    fields_3d = {field: start_time for field in fields}

    # 2D pressure at top of domain.
    endtime = (domain.end_date - domain.start_date).total_seconds()
    hours = [start_time + n*3600 for n in range(int(endtime // 3600) + 1)]

    fields_2d = {
            'phydro_tod': hours}

    regrid_les(
            fields_3d,
            fields_2d,
            domain.parent.xsize,
            domain.parent.ysize,
            gd['z'],
            gd['zh'],
            domain.parent.itot,
            domain.parent.jtot,
            domain.xsize,
            domain.ysize,
            gd['z'],
            gd['zh'],
            domain.itot,
            domain.jtot,
            domain.xstart_in_parent,
            domain.ystart_in_parent,
            domain.parent.work_dir,
            domain.work_dir,
            time_offset,
            float_type=settings.float_type,
            name_suffix='ext')

    # Link boundary conditions from parent to child domain.
    link_wtop = True

    link_bcs_from_parent(
        fields,
        start_time,
        end_time,
        domain.lbc_freq,
        link_wtop,
        domain.parent.work_dir,
        domain.work_dir,
        time_offset)

    link_buffer_from_parent(
        ['u', 'v', 'w', 'thl', 'qt'],
        start_time,
        end_time,
        domain.buffer_freq,
        domain.parent.work_dir,
        domain.work_dir,
        time_offset)


def create_nc_input(era5_1d, era5_1d_mean, domain, case_name):
    """
    Create `case_input.nc` file.
    """
    logger.info(f'Creating {case_name}_input.nc')

    # RFMIP concentrations as background species for RTE+RRTMGP.
    lon = domain.proj.central_lon
    lat = domain.proj.central_lat
    rfmip = get_rfmip_species(lat, lon, exp=0)

    # No need to init the chemical species, the initial 3D fields
    # are overwritten by the fields interpolated from CAMS.
    # The same is true for all fields, but at least leave some
    # meteorology in here for testing with periodic BCs.
    eps = xm_cams['h2o'] / xm_cams['air']
    h2o = era5_1d['qt'][0,:] / (eps - eps * era5_1d['qt'][0,:])

    init_profiles = {
            'z': settings.vgrid.z,
            'thl': era5_1d['thl'][0,:],
            'qt': era5_1d['qt'][0,:],
            'u': era5_1d['u'][0,:],
            'v': era5_1d['v'][0,:],
            'o3': era5_1d['o3'][0,:]*1e-6,
            'h2o': h2o}

    radiation  = {
        'z_lay': era5_1d_mean['z_lay'  ],
        'z_lev': era5_1d_mean['z_lev'  ],
        'p_lay': era5_1d_mean['p_lay'  ],
        'p_lev': era5_1d_mean['p_lev'  ],
        't_lay': era5_1d_mean['t_lay'  ],
        't_lev': era5_1d_mean['t_lev'  ],
        'o3':    era5_1d_mean['o3_lay' ]*1e-6,
        'h2o':   era5_1d_mean['h2o_lay']}

    for name, conc in rfmip.items():
        init_profiles[name] = conc
        radiation[name] = conc

    # Large-scale forcings (not always used...).
    tdep_ls = {
        'time_ls': era5_1d.time_sec,
        'u_geo' : era5_1d['ug'],
        'v_geo' : era5_1d['vg']}

    # Save in NetCDF format.
    save_case_input(
        case_name = case_name,
        init_profiles = init_profiles,
        radiation = radiation,
        tdep_ls = tdep_ls,
        output_dir = domain.work_dir)


def create_ini(domain, era5_1d, case_name):
    """
    Read base .ini file and fill in details.
    """
    logger.info(f'Creating {case_name}.ini')

    child = domain.child
    parent = domain.parent

    ini = read_ini(f'{case_name}.ini.base')

    ini['master']['npx'] = domain.npx
    ini['master']['npy'] = domain.npy

    ini['grid']['itot'] = domain.itot
    ini['grid']['jtot'] = domain.jtot
    ini['grid']['ktot'] = settings.vgrid.kmax

    ini['grid']['xsize'] = domain.xsize
    ini['grid']['ysize'] = domain.ysize
    ini['grid']['zsize'] = settings.vgrid.zsize

    ini['grid']['lat'] = domain.proj.central_lat
    ini['grid']['lon'] = domain.proj.central_lon

    ini['buffer']['zstart'] = 0.75 * settings.vgrid.zsize
    ini['buffer']['loadfreq'] = domain.buffer_freq

    if settings.sw_ls == '1d_geo':
        # Domain mean geostrophic wind and single Coriolis parameter.
        ini['force']['swlspres'] = 'geo'
        ini['force']['swtimedep_geo'] = True
        ini['force']['swrotation_2d'] = False
        ini['force']['fc'] = era5_1d.fc
    elif settings.sw_ls == '3d_geo':
        # 3D geostrophic wind, uses 2D Coriolis parameter by default.
        ini['force']['swlspres'] = 'geo3d'
        ini['force']['swtimedep_geo'] = True
        ini['force']['ugeo_loadtime'] = 3600
        ini['force']['swrotation_2d'] = False
        ini['force']['fc'] = -1
    elif settings.sw_ls == 'no_geo':
        # No geostrophic wind, but still use 2D rotation.
        ini['force']['swlspres'] = '0'
        ini['force']['swtimedep_geo'] = False
        ini['force']['swrotation_2d'] = True
        ini['force']['fc'] = -1

    ini['time']['endtime'] = (domain.end_date - domain.start_date).total_seconds()
    ini['time']['datetime_utc'] = domain.start_date.strftime('%Y-%m-%d %H:%M:%S')

    ini['boundary_lateral']['loadfreq'] = domain.lbc_freq
    ini['boundary_lateral']['n_sponge'] = domain.n_sponge
    ini['boundary_lateral']['tau_sponge'] = domain.n_sponge * domain.dx / (10 * 5)

    ini['cross']['xz'] = domain.ysize/2
    ini['cross']['yz'] = domain.xsize/2

    x,y = domain.proj.to_xy(settings.lon_mmij, settings.lat_mmij)
    ini['column']['coordinates[x]'] = x
    ini['column']['coordinates[y]'] = y

    if parent is None:
        ini['radiation']['dt_rad'] = 300
    else:
        ini['radiation']['dt_rad'] = 60

    if child is not None:
        ini['subdomain']['sw_subdomain'] = True

        ini['subdomain']['xstart'] = child.xstart_in_parent
        ini['subdomain']['ystart'] = child.ystart_in_parent
        ini['subdomain']['xend'] = child.xstart_in_parent + child.xsize
        ini['subdomain']['yend'] = child.ystart_in_parent + child.ysize

        ini['subdomain']['grid_ratio_ij'] = int(domain.dx / child.dx)
        ini['subdomain']['grid_ratio_k'] = 1
        ini['subdomain']['n_ghost'] = child.n_ghost
        ini['subdomain']['n_sponge'] = child.n_sponge

        ini['subdomain']['sw_save_wtop'] = True
        ini['subdomain']['sw_save_buffer'] = True

        ini['subdomain']['savetime_bcs'] = child.lbc_freq
        ini['subdomain']['savetime_buffer'] = child.buffer_freq
        ini['subdomain']['zstart_buffer'] = settings.zstart_buffer
    else:
        ini['subdomain']['sw_subdomain'] = False

    # Check if all None values are set.
    check_ini(ini)

    # Write to output .ini file.
    save_ini(ini, f'{domain.work_dir}/{case_name}.ini')


def create_surface_input(era5, domain, bs):
    """
    Create surface input from SST.
    """
    logger.info(f'Creating SSTs')

    # Create SSTs from ERA5.
    sst_les = create_sst_from_regular_latlon(
        era5.sst[0],
        era5.lons,
        era5.lats,
        domain.proj.lon,
        domain.proj.lat,
        extrapolate_sea=True,
        float_type=settings.float_type)

    if np.any(np.isnan(sst_les)):
        logger.critical('SSTs contain NaNs!')

    p_bot = bs['ph'][0]
    qsat_bot = qsat(p_bot, sst_les)
    thl_bot = sst_les / exner(p_bot)

    qsat_bot.tofile(f'{domain.work_dir}/qt_bot_in.0000000')
    thl_bot.tofile(f'{domain.work_dir}/thl_bot_in.0000000')


def create_ls_and_rotation(domain, vgrid, era5):
    """
    Create large-scale forcings such as geostrophic wind, 2D Coriolis frequency, ...
    """
    time = era5.time_sec.astype(np.int32)

    # 2D Coriolis frequency.
    fc_2d = create_2d_coriolis_freq(domain.proj.lat, settings.float_type)
    fc_2d.tofile(f'{domain.work_dir}/fc.0000000')

    if settings.sw_ls == '3d_geo':
        # 3D Geostrophic wind. Files are written directly to `domain.work_dir`.
        create_3d_geowind_from_regular_latlon(
            era5.ug,
            era5.vg,
            era5.lons.data,
            era5.lats.data,
            era5.z,
            time,
            vgrid.z,
            domain,
            domain.work_dir,
            ntasks=8,
            float_type=settings.float_type)


def copy_lookup_tables(env, domain):
    """
    Copy required land-surface and radiation lookup tables.
    """
    logger.info(f'Copying lookup tables')

    microhh_path = env['microhh_path']
    gpt_path = env['gpt_path']

    rrtmgp_path = f'{microhh_path}/rte-rrtmgp-cpp/'
    rrtmgp_data_path = f'{microhh_path}/rte-rrtmgp-cpp/rrtmgp-data'

    to_copy = [
            (f'{gpt_path}/rrtmgp-gas-lw-g056-cf2.nc', 'coefficients_lw.nc'),
            (f'{gpt_path}/rrtmgp-gas-sw-g049-cf2.nc', 'coefficients_sw.nc'),
            (f'{rrtmgp_data_path}/rrtmgp-clouds-lw.nc', 'cloud_coefficients_lw.nc'),
            (f'{rrtmgp_data_path}/rrtmgp-clouds-sw.nc', 'cloud_coefficients_sw.nc'),
            (f'{rrtmgp_path}/data/aerosol_optics.nc', 'aerosol_optics.nc'),
            (f'{microhh_path}/misc/van_genuchten_parameters.nc', 'van_genuchten_parameters.nc')]

    for f in to_copy:
        target = f'{domain.work_dir}/{f[1]}'
        if not os.path.exists(target):
            shutil.copy(f[0], target)


#def main():
if True:

    # Short-cuts
    case_name = settings.ls2d_settings['case_name']

    # Parse command line arguments.
    args = parse_args()

    # Switch between domain specification.
    domain = settings.outer_dom if args.domain == 'outer' else settings.inner_dom

    # Create work directory.
    if not os.path.exists(domain.work_dir):
        os.makedirs(domain.work_dir)

    # Initial / boundary conditions from ERA5 / CAMS using (LS)2D.
    era5_3d, era5_1d, era5_1d_mean = read_era5_cams(
        settings.ls2d_settings,
        domain.start_date,
        domain.end_date,
        settings.vgrid)

    if args.domain == 'outer':
        # Create base state density.
        bs = create_basestate(era5_1d_mean, settings.vgrid)

        # Create initial fields and lateral/top boundary conditions.
        create_init_and_bcs_outer(era5_3d, domain, bs)

    else:
        # Read base state density from parent. They *must* be identical.
        bs = read_moist_basestate(f'{domain.parent.work_dir}/thermo_basestate.0000000')

        # Interpolate / link fields from parent to child domain.
        create_init_and_bcs_inner(domain, bs, settings.vgrid)

    # Save both basestate density and entire moist thermo basestate.
    save_basestate_density(bs['rho'], bs['rhoh'], f'{domain.work_dir}/rhoref_ext.0000000')
    save_moist_basestate(bs, f'{domain.work_dir}/thermo_basestate_ext.0000000')

    # Create `case_input.nc` NetCDF file.
    create_nc_input(era5_1d, era5_1d_mean, domain, case_name)

    # Create `case.ini` from `case.ini.base`, filling in details.
    create_ini(domain, era5_1d, case_name)

    # Create land-surface (vegetation) and sea (SST) input.
    create_surface_input(era5_3d, domain, bs)

    # Create large-scale forcings.
    create_ls_and_rotation(domain, settings.vgrid, era5_3d)

    # Copy surface and radiation lookup tables.
    copy_lookup_tables(settings.env, domain)


#if __name__ == '__main__':
#    main()
