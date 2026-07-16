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

import shutil
import os

import numpy as np

# pip install ls2d
import ls2d

# pip install microhhpy
from microhhpy.land import create_land_surface_input, Land_surface_input
from microhhpy.io import read_ini, check_ini, save_ini, save_case_input
from microhhpy.chem import get_rfmip_species
from microhhpy.logger import logger
from microhhpy.constants import xm_cams

# Local settings and scripts.
from global_settings import float_type, ls2d_settings, env, domain, vgrid
from global_settings import cams_egg4_variables, chemical_species
from global_settings import stacks
from stack_properties import get_source


def read_era5_cams(ls2d_settings, start_date, end_date, cams_variables, vgrid):
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

    # Read CAMS data.
    cams = ls2d.Read_cams(ls2d_settings, cams_variables)

    # Interpolate to LES grid.
    cams_les = cams.get_les_input(vgrid.z)

    return era5, era5_les, era5_mean, cams, cams_les


def create_nc_input(era5_1d, era5_1d_mean, cams_1d, chemical_species, domain, case_name):
    """
    Create `case_input.nc` file.
    """
    logger.info(f'Creating {case_name}_input.nc')

    # RFMIP concentrations as background species for RTE+RRTMGP.
    lon = domain.proj.central_lon
    lat = domain.proj.central_lat
    rfmip = get_rfmip_species(lat, lon, exp=0)

    # Scalar fields from CAMS. Input CAMS = mass mixing ratio (kg/kg), convert to volume mixing ratio.
    # NOTE: CO2 only in this case. Leave in mass mixing ratio!
    species_cams = {}
    for specie in chemical_species:
        species_cams[specie] = cams_1d[specie].values # * xm_cams['air'] / xm_cams[specie]

    eps = xm_cams['h2o'] / xm_cams['air']
    h2o = era5_1d['qt'][0,:] / (eps - eps * era5_1d['qt'][0,:])
    nudgefac = np.ones(vgrid.kmax) / 10800      # s-1

    init_profiles = {
            'z': vgrid.z,
            'thl': era5_1d['thl'][0,:],
            'qt': era5_1d['qt'][0,:],
            'u': era5_1d['u'][0,:],
            'v': era5_1d['v'][0,:],
            'o3': era5_1d['o3'][0,:]*1e-6,
            'h2o': h2o,
            'nudgefac': nudgefac}

    for name, conc in species_cams.items():
        init_profiles[name] = conc[0,:]

    radiation  = {
        'z_lay': era5_1d_mean['z_lay'  ],
        'z_lev': era5_1d_mean['z_lev'  ],
        'p_lay': era5_1d_mean['p_lay'  ],
        'p_lev': era5_1d_mean['p_lev'  ],
        't_lay': era5_1d_mean['t_lay'  ],
        't_lev': era5_1d_mean['t_lev'  ],
        'o3':    era5_1d_mean['o3_lay' ]*1e-6,
        'h2o':   era5_1d_mean['h2o_lay']}

    # NOTE: not used with heterogeneous surface, but still required by MicroHH.
    soil_index = int(era5_1d.type_soil-1)  # -1 = Fortran -> C indexing
    soil = {
            'z': era5_1d.zs[::-1],
            'theta_soil': era5_1d.theta_soil[0,::-1],
            't_soil': era5_1d.t_soil[0,::-1],
            'index_soil': np.ones(4) * soil_index,
            'root_frac': era5_1d.root_frac_low_veg[::-1]}

    for name, conc in rfmip.items():
        if name not in species_cams:
            init_profiles[name] = conc
        radiation[name] = conc

    # Large-scale forcings and inflow.
    tdep_ls = {
        'time_ls': era5_1d.time_sec,
        'u_geo' : era5_1d.ug,
        'v_geo' : era5_1d.vg,
        'thl_nudge' : era5_1d.thl,
        'qt_nudge' : era5_1d.qt,
        'u_nudge' : era5_1d.u,
        'v_nudge' : era5_1d.v,
        'thl_ls' : era5_1d.dtthl_advec,
        'qt_ls' : era5_1d.dtqt_advec,
        'u_ls' : era5_1d.dtu_advec,
        'v_ls' : era5_1d.dtv_advec,
        'w_ls' : era5_1d.wls}

    for name, conc in species_cams.items():
        tdep_ls[f'{name}_nudge'] = conc
        tdep_ls[f'{name}_inflow'] = conc

    # Save in NetCDF format.
    save_case_input(
        case_name = case_name,
        init_profiles = init_profiles,
        radiation = radiation,
        soil = soil,
        tdep_ls = tdep_ls,
        output_dir = domain.work_dir)


def create_ini(domain, era5_1d, species, case_name):
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
    ini['grid']['ktot'] = vgrid.kmax

    ini['grid']['xsize'] = domain.xsize
    ini['grid']['ysize'] = domain.ysize
    ini['grid']['zsize'] = vgrid.zsize

    ini['grid']['lat'] = domain.proj.central_lat
    ini['grid']['lon'] = domain.proj.central_lon

    ini['buffer']['zstart'] = 0.75 * vgrid.zsize

    ini['boundary']['scalar_outflow'] = species

    ini['force']['fc'] = era5_1d.fc
    ini['force']['nudgelist'] = ['thl', 'qt', 'u', 'v'] + species
    ini['force']['timedeplist_nudge'] = ['thl', 'qt', 'u', 'v'] + species

    ini['fields']['slist'] = species
    ini['advec']['fluxlimit_list'] = ['qt'] + species
    ini['limiter']['limitlist'] = ['qt'] + species

    ini['time']['endtime'] = (domain.end_date - domain.start_date).total_seconds()
    ini['time']['datetime_utc'] = domain.start_date.strftime('%Y-%m-%d %H:%M:%S')

    ini['cross']['crosslist'] += ['co2', 'co2_path']
    ini['cross']['xz'] = domain.ysize/2
    ini['cross']['yz'] = domain.xsize/2

    # Check if all None values are set.
    check_ini(ini)

    # Write to output .ini file.
    save_ini(ini, f'{domain.work_dir}/{case_name}.ini')


def create_surface_input(era5_mean, domain, env):
    """
    Create land-surface (vegetation) and sea (SST) input.
    """
    logger.info(f'Creating spatial (land-) surface input')

    # Default soil depths IFS.
    z_soil = np.array([-0.035, -0.175, -0.64 , -1.945])[::-1]

    # Land-surface / vegetation properties from global LCC dataset (100 m resolution).
    lu_lcc = create_land_surface_input(
        domain.proj.lon,
        domain.proj.lat,
        z_soil,
        land_use_source='corine_100m',
        land_use_tiff=env['corine_path'],
        save_binaries=True,
        output_dir=domain.work_dir,
        save_netcdf=True,
        netcdf_file='lsm_input.nc',
        float_type=float_type
    )

    # TODO: Init soil from HiHydroSoil. For now spatially homogeneous.
    soil = Land_surface_input(
        domain.itot,
        domain.jtot,
        4,
        exclude_veg=True,
        debug=True,
        float_type=float_type
    )

    soil.theta_soil[:,:,:] = era5_mean.theta_soil.values[::-1, None, None]
    soil.t_soil[:,:,:] = era5_mean.t_soil.values[::-1, None, None]
    soil.index_soil[:,:,:] = int(era5_mean.type_soil) - 1  # FORTRAN -> C
    soil.to_binaries(path=domain.work_dir, allow_overwrite=True)


def create_emissions(stacks, domain):
    """
    Create 3D emission input.
    """
    pass


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

    # Create work directory.
    if not os.path.exists(domain.work_dir):
        os.makedirs(domain.work_dir)

    # Initial / boundary conditions from ERA5 / CAMS using (LS)2D.
    _, era5_1d, era5_1d_mean, _, cams_1d = read_era5_cams(
        ls2d_settings, domain.start_date, domain.end_date, cams_egg4_variables, vgrid)

    # Create `case_input.nc` NetCDF file.
    create_nc_input(era5_1d, era5_1d_mean, cams_1d, chemical_species, domain, ls2d_settings['case_name'])

    # Create `case.ini` from `case.ini.base`, filling in details.
    create_ini(domain, era5_1d, chemical_species, ls2d_settings['case_name'])

    # Create land-surface (vegetation) and sea (SST) input.
    create_surface_input(era5_1d_mean, domain, env)

    # Copy surface and radiation lookup tables.
    copy_lookup_tables(env, domain)


#if __name__ == '__main__':
#    main()
