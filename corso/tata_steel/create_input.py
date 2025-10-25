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

import pandas as pd
import numpy as np

# pip install ls2d
import ls2d

# pip install microhhpy
from microhhpy.real import create_input_from_regular_latlon
from microhhpy.real import create_sst_from_regular_latlon
from microhhpy.real import regrid_les, link_bcs_from_parent, link_buffer_from_parent
from microhhpy.land import create_land_surface_input, Land_surface_input
from microhhpy.thermo import calc_moist_basestate, save_basestate_density, read_basestate_density
from microhhpy.io import read_ini, check_ini, save_ini, save_case_input
from microhhpy.chem import get_rfmip_species, fit_gaussian_curve
from microhhpy.spatial import calc_vertical_grid_2nd
from microhhpy.chem import calc_tuv_photolysis
from microhhpy.utils import get_data_file
from microhhpy.logger import logger
from microhhpy.constants import xm_cams

# Local settings and scripts.
from global_settings import sw_openbc, sw_scalars, sw_chemistry
from global_settings import float_type, ls2d_settings, env, outer_dom, inner_dom, vgrid, zstart_buffer
from global_settings import cams_eac4_variables, chemical_species, lumping_species
from corso_emissions import Corso_emissions


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
    cams.ds_ml = cams.ds_ml.rename({'go3': 'o3'})

    return era5, era5_les, era5_mean, cams


def create_basestate(era5_mean, vgrid, domain):
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
        float_type)

    return bs


def create_init_and_bcs_outer(era5, cams, domain, bs):
    """
    Create the initial 3D fields interpolated from ERA5/CAMS,
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

    # Meteorological fields.
    create_input_from_regular_latlon(
        fields_era,
        era5.lons.data,   # Strip off array masks.
        era5.lats.data,
        era5.z[:,:,:,:],
        era5.p[:,:,:,:],
        time_sec,
        vgrid.z,
        vgrid.zsize,
        zstart_buffer,
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
        float_type=float_type)

    if sw_scalars:

        # Scalar fields from CAMS. Input CAMS = mass mixing ratio (kg/kg), convert to volume mixing ratio.
        species_cams = {}
        for specie in chemical_species:
            if specie == 'co2':
                logger.warning('CO2 not available, setting to 420 ppm.')
                species_cams['co2'] = np.ones_like(cams.ds_ml['no2']) * 420e-6
            elif specie not in lumping_species:
                species_cams[specie] = cams.ds_ml[specie].values * xm_cams['air'] / xm_cams[specie]

        # Sum lumped species as sum of converted volume mixing ratios.
        for output_specie, sub_species in lumping_species.items():
            species_cams[output_specie] = np.zeros_like(species_cams['co'])
            for sub_specie in sub_species:
                species_cams[output_specie] += cams.ds_ml[sub_specie].values * xm_cams['air'] / xm_cams[sub_specie]

        create_input_from_regular_latlon(
            species_cams,
            cams.ds_ml.longitude.values,
            cams.ds_ml.latitude.values,
            cams.ds_ml.z.values,
            None,
            time_sec,
            vgrid.z,
            vgrid.zsize,
            zstart_buffer,
            bs['rho'],
            bs['rhoh'],
            domain,
            sigma_h,
            clip_at_zero=list(species_cams.keys()),
            name_suffix='ext',
            output_dir=domain.work_dir,
            ntasks=8,
            float_type=float_type)


def init_and_bcs_inner(domain, bs, vgrid):

    gd = calc_vertical_grid_2nd(vgrid.z, vgrid.zsize, float_type=float)

    # Create and/or link initial fields and boundary conditions from parent domain.
    start_time = int((domain.start_date - domain.parent.start_date).total_seconds())
    end_time = int(start_time + (domain.end_date - domain.start_date).total_seconds())
    time_offset = -start_time

    # Initial 3D fields at (optionally) delayed start time of child domain.
    fields = ['u', 'v', 'w', 'thl', 'qt'] + chemical_species
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
            float_type=float_type,
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


def create_nc_input(era5_1d, era5_1d_mean, df_tuv, emissions, domain, case_name):
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
            'z': vgrid.z,
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

    # NOTE: not used with heterogeneous surface, but still required by MicroHH.
    soil_index = int(era5_1d.type_soil-1)  # -1 = Fortran -> C indexing
    soil = {
            'z': era5_1d.zs[::-1],
            'theta_soil': era5_1d.theta_soil[0,::-1],
            't_soil': era5_1d.t_soil[0,::-1],
            'index_soil': np.ones(4) * soil_index,
            'root_frac': era5_1d.root_frac_low_veg[::-1]}

    for name, conc in rfmip.items():
        init_profiles[name] = conc
        radiation[name] = conc

    # Photolysis rates.
    if sw_chemistry:
        time_chem = ((df_tuv.index - df_tuv.index[0]).values * 1e-9).astype('float64')
        emi_isop = np.zeros_like(time_chem)
        emi_no = np.zeros_like(time_chem)

        tdep_chem = {
            'time_chem': time_chem,
            'jo31d': df_tuv.jo31d,
            'jh2o2': df_tuv.jh2o2,
            'jno2': df_tuv.jno2,
            'jno3': df_tuv.jno3,
            'jn2o5': df_tuv.jn2o5,
            'jch2or': df_tuv.jch2or,
            'jch2om': df_tuv.jch2om,
            'jch3o2h': df_tuv.jch3o2h,
            'emi_isop': emi_isop,
            'emi_no': emi_no,
        }
    else:
        tdep_chem = None

    # Time dependent emissions.
    endtime = (domain.end_date - domain.start_date).total_seconds()

    tdep_source = {'time_source': np.arange(0, endtime+1, 3600)}
    for n,e in enumerate(emissions):
        tdep_source[f'source_strength_{n}'] = e['strength']

    # Save in NetCDF format.
    save_case_input(
        case_name = case_name,
        init_profiles = init_profiles,
        radiation = radiation,
        soil = soil,
        tdep_chem = tdep_chem,
        tdep_source = tdep_source,
        output_dir = domain.work_dir)


def create_ini(domain, emissions, case_name):
    """
    Read base .ini file and fill in details.
    """
    logger.info(f'Creating {case_name}.ini')

    child = domain.child
    parent = domain.parent

    scalars = chemical_species if sw_scalars else []

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
    ini['buffer']['loadfreq'] = domain.buffer_freq

    ini['chemistry']['swchemistry'] = sw_chemistry
    ini['deposition']['swdeposition'] = sw_chemistry

    if sw_scalars:
        ini['fields']['slist'] = scalars
    ini['advec']['fluxlimit_list'] = ['qt'] + scalars
    ini['limiter']['limitlist'] = ['qt'] + scalars

    ini['time']['endtime'] = (domain.end_date - domain.start_date).total_seconds()
    ini['time']['datetime_utc'] = domain.start_date.strftime('%Y-%m-%d %H:%M:%S')

    # Emissions from lat/lon to x/y.
    for e in emissions:
        e['x'], e['y'] = domain.proj.to_xy(e['lon'], e['lat'])

    ini['source']['swsource'] = sw_scalars
    if sw_scalars:
        ini['source']['sourcelist'] = [e['specie'] for e in emissions]
        ini['source']['source_x0'] = [e['x'] for e in emissions]
        ini['source']['source_y0'] = [e['y'] for e in emissions]
        ini['source']['source_z0'] = [e['z'] for e in emissions]
        ini['source']['sigma_x'] = [e['sigma_x'] for e in emissions]
        ini['source']['sigma_y'] = [e['sigma_y'] for e in emissions]
        ini['source']['sigma_z'] = [e['sigma_z'] for e in emissions]
        ini['source']['strength'] = len(emissions)*[-1]
        #ini['source']['strength'] = [e['strength'] for e in emissions]
        ini['source']['swvmr'] = [True for e in emissions]
        ini['source']['line_x'] = [0 for e in emissions]
        ini['source']['line_y'] = [0 for e in emissions]
        ini['source']['line_z'] = [0 for e in emissions]

    ini['boundary_lateral']['loadfreq'] = domain.lbc_freq
    ini['boundary_lateral']['n_sponge'] = domain.n_sponge
    ini['boundary_lateral']['tau_sponge'] = domain.n_sponge * domain.dx / (10 * 5)
    ini['boundary_lateral']['slist'] = ['thl', 'qt'] + scalars

    if sw_scalars:
        chem_vars = ['no', 'no2', 'o3', 'co', 'oh', 'co2']
        path_vars = ['no_path', 'no2_path', 'o3_path', 'co_path', 'oh_path', 'co2_path']
        ini['cross']['crosslist'] += chem_vars + path_vars

    ini['cross']['xz'] = domain.ysize/2
    ini['cross']['yz'] = domain.xsize/2

    ini['pres']['sw_openbc'] = sw_openbc
    ini['boundary_lateral']['sw_openbc'] = sw_openbc

    if parent is None:
        ini['boundary_lateral']['sw_recycle'] = True
        ini['radiation']['dt_rad'] = 300
    else:
        ini['boundary_lateral']['sw_recycle'] = False
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
        ini['subdomain']['zstart_buffer'] = zstart_buffer
    else:
        ini['subdomain']['sw_subdomain'] = False

    # Check if all None values are set.
    check_ini(ini)

    # Write to output .ini file.
    save_ini(ini, f'{domain.work_dir}/{case_name}.ini')


def create_surface_input(era5, era5_mean, domain, env):
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
        land_use_source='lcc_100m',
        land_use_tiff=env['lcc_path'],
        save_binaries=True,
        output_dir=domain.work_dir,
        save_netcdf=True,
        netcdf_file='lsm_input.nc')


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

    # Create SSTs from ERA5.
    sst_les = create_sst_from_regular_latlon(
        era5.sst[0],
        era5.lons,
        era5.lats,
        domain.proj.lon,
        domain.proj.lat,
        float_type=float_type)

    if np.any(np.isnan(sst_les)):
        logger.warning('SSTs contain NaNs! Setting to 290K...')
        sst_les[:,:] = 290.

    # We don't know water temperatures over land, and the extrapolated SSTs are of course not a very accurate estimation...
    # TODO: get inland water mask from Corine/LCC and let user define water/lake temperatures?
    sst_les[sst_les < 280] = 280
    sst_les.tofile(f'{domain.work_dir}/t_bot_water.0000000')


def calc_photolysis_rates(env, domain):
    """
    Calculate photolysis rates using TUV wrapper.
    """
    name = 'microh'   # Must be exactly 6 characters!

    # Default input file. Only start/end date and lat/lon location are updated.
    input_file = get_data_file('microhh_tuv.base')

    tuv_df = calc_tuv_photolysis(
            input_file,
            env['tuv_path'],
            name,
            domain.start_date,
            domain.end_date,
            domain.proj.central_lon,
            domain.proj.central_lat,
            suppress_stdout=True)

    return tuv_df


def create_emissions(chemical_species, domain, env, sigma_x=100, sigma_y=100, no_no2_ratio=0.95):
    """
    Get all time varying emissions from CORSO catalogue within domain.
    """

    dates = pd.date_range(domain.start_date, domain.end_date, freq='1h')

    e = Corso_emissions(env['corso_path'])

    margin = 0.025    # ~2.5 km

    min_lon = domain.proj.lon.min() + margin
    max_lon = domain.proj.lon.max() - margin

    min_lat = domain.proj.lat.min() + margin
    max_lat = domain.proj.lat.max() - margin

    e.filter_emissions(min_lon, max_lon, min_lat, max_lat)

    # Gather emission info in list of dicts, to pass
    # to `create_ini()` and `create_nc_input()` functions.
    emissions = []

    for index, row in e.df_emiss.iterrows():

        # Get vertical distribution and fit Gaussian curve.
        # MicroHH currently does not support vertical profiles
        # combined with time varying strength.
        z_emiss, p_emiss = e.get_profile(index)
        curve_fit = fit_gaussian_curve(z_emiss, p_emiss)

        def add_emission(specie, strength):
            # Input = kg/s, output for KPP = kmol/s.
            strength /= xm_cams[specie]

            emissions.append(
                dict(
                    specie=specie,
                    lat=row.latitude,
                    lon=row.longitude,
                    z=curve_fit['x0'],
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    sigma_z=curve_fit['sigma'],
                    strength=strength
                ))

        # Only include species that are used.
        species = []
        if 'co2' in chemical_species:
            species.append('co2')
        if 'no' in chemical_species and 'no2' in chemical_species:
            species.append('nox')
        if 'co' in chemical_species:
            species.append('co')

        # Get emissions as function of time for each specie.
        for specie in species:

            strength = e.get_emission_tser(index, specie, dates)    # Units `kg/s`.

            if specie == 'nox':
                # Split NOx in to NO and NO2.
                no_strength = strength * no_no2_ratio * (xm_cams['no'] / xm_cams['no2'])
                no2_strength = strength * (1 - no_no2_ratio)

                add_emission('no', no_strength)
                add_emission('no2', no2_strength)

            else:
                add_emission(specie, strength)

    logger.info(f'Found {len(e.df_emiss)} emission(s) in domain.')

    return emissions


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


def main():

    # Parse command line arguments.
    args = parse_args()

    # Switch between domain specification.
    domain = outer_dom if args.domain == 'outer' else inner_dom

    # Create work directory.
    if not os.path.exists(domain.work_dir):
        os.makedirs(domain.work_dir)

    # Initial / boundary conditions from ERA5 / CAMS using (LS)2D.
    era5_3d, era5_1d, era5_1d_mean, cams_3d = read_era5_cams(
        ls2d_settings, domain.start_date, domain.end_date, cams_eac4_variables, vgrid)

    if args.domain == 'outer':
        # Create base state density.
        bs = create_basestate(era5_1d_mean, vgrid, domain)

        # Create initial fields and lateral/top boundary conditions.
        create_init_and_bcs_outer(era5_3d, cams_3d, domain, bs)

    else:
        # Read base state density from parent. They *must* be identical.
        bs = read_basestate_density(f'{domain.parent.work_dir}/rhoref.0000000')

        # Interpolate / link fields from parent to child domain.
        init_and_bcs_inner(domain, bs, vgrid)

    # Save basestate density.
    save_basestate_density(bs['rho'], bs['rhoh'], f'{domain.work_dir}/rhoref_ext.0000000')

    # Setup emissions from corso_ps_catalogue_v2.0 and corso_ps_* time/vertical profiles.
    emissions = create_emissions(chemical_species, domain, env, sigma_x=100, sigma_y=100, no_no2_ratio=0.95)

    # Calculate photolysis rates for KPP
    if sw_chemistry:
        df_tuv = calc_photolysis_rates(env, domain)
    else:
        df_tuv = None

    # Create `case_input.nc` NetCDF file.
    create_nc_input(era5_1d, era5_1d_mean, df_tuv, emissions, domain, ls2d_settings['case_name'])

    # Create `case.ini` from `case.ini.base`, filling in details.
    create_ini(domain, emissions, ls2d_settings['case_name'])

    # Create land-surface (vegetation) and sea (SST) input.
    create_surface_input(era5_3d, era5_1d_mean, domain, env)

    # Copy surface and radiation lookup tables.
    copy_lookup_tables(env, domain)


if __name__ == '__main__':
    main()
