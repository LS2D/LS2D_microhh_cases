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


import numpy as np

from microhhpy.thermo import qsat
import microhhpy.constants as cst


rho_std = 1.29      # flue gas density at standard state   [kg m-3]
cp_vol  = 1.36e-3   # volumetric heat capacity of flue gas [MW s m-3 K-1], Pregger Eq. (1)


"""
Stack parameters for large combustion plants, digitized from:
Pregger & Friedrich (2009), Env. Pollution 157, 552-560, Fig. 1.
Bins are thermal capacity (min_MWth, max_MWth); max = None is open-ended.
  h     : stack height                        [m]
  T     : flue gas temperature                [K]
  w     : flue gas exit velocity              [m s-1], at stack conditions
  V_std : flue gas flow rate, standard state  [m3 s-1]
"""
stack_properties = {
    'lignite': [
        {'range': (50, 100),    'h':  80, 'T': 448.15, 'w': 2.7,  'V_std':  20},
        {'range': (100, 300),   'h': 120, 'T': 441.15, 'w': 3.8,  'V_std':  55},
        {'range': (300, 1000),  'h': 160, 'T': 418.15, 'w': 5.8,  'V_std': 150},
        {'range': (1000, None), 'h': 250, 'T': 423.15, 'w': 3.6,  'V_std': 340},
        ],
    'lignite_cooling_tower': [
        {'range': (300, 1000),  'h': 110, 'T': 305.15, 'w': 0.15, 'V_std': 425},
        {'range': (1000, None), 'h': 120, 'T': 303.15, 'w': 0.15, 'V_std': 785},
        ],
    'hard_coal': [
        {'range': (50, 100),    'h':  80, 'T': 413.15, 'w': 4.2,  'V_std':  10},
        {'range': (100, 300),   'h': 120, 'T': 413.15, 'w': 5.6,  'V_std':  32},
        {'range': (300, 1000),  'h': 160, 'T': 383.15, 'w': 7.5,  'V_std': 115},
        {'range': (1000, None), 'h': 235, 'T': 376.15, 'w': 9.7,  'V_std': 360},
        ],
    'natural_gas': [
        {'range': (50, 100),    'h':  55, 'T': 411.15, 'w': 4.4,  'V_std':  10},
        {'range': (100, 300),   'h': 100, 'T': 408.15, 'w': 6.1,  'V_std':  30},
        {'range': (300, 600),   'h': 115, 'T': 401.15, 'w': 5.9,  'V_std':  72},
        {'range': (600, None),  'h': 170, 'T': 403.15, 'w': 5.7,  'V_std':  98},
        ],
    'light_fuel_oil': [
        {'range': (50, 100),    'h':  60, 'T': 491.15, 'w': 3.9,  'V_std':  18},
        {'range': (100, 300),   'h': 120, 'T': 436.15, 'w': 3.1,  'V_std':  22},
        {'range': (300, None),  'h': 160, 'T': 421.15, 'w': 6.7,  'V_std':  58},
        ],
    'heavy_fuel_oil': [
        {'range': (50, 100),    'h':  90, 'T': 453.15, 'w': 3.7,  'V_std':  10},
        {'range': (100, 300),   'h': 115, 'T': 425.15, 'w': 4.1,  'V_std':  20},
        {'range': (300, None),  'h': 155, 'T': 470.15, 'w': 2.2,  'V_std':  42},
        ],
    }


def lookup_stacks(fuel, q_th):
    """
    Pregger bin lookup by thermal capacity.

    Arguments:
        fuel       : key of `properties`
        q_th       : thermal capacity [MW_th]
        properties : LUT of stack properties per fuel and capacity bin

    Returns dict with h [m], T [K], w [m s-1], V_std [m3 s-1].
    """

    if fuel not in stack_properties:
        raise KeyError(f'unknown fuel "{fuel}".')

    for entry in stack_properties[fuel]:
        lo, hi = entry['range']
        if q_th >= lo and (hi is None or q_th < hi):
            return {k: v for k, v in entry.items() if k != 'range'}

    raise ValueError(f'q_th={q_th:.1f} MW_th outside tabulated range for "{fuel}".')


def get_source(site, p_env):
    """
    Emission parameters for one source.

    Arguments:
        site  : dict with fuel, p, eta, rh, h, lat, lon
        p_env : base state pressure at outlet height [Pa]

    Returns dict with:
        lat, lon : location [deg]
        h        : outlet height [m]
        d        : outlet diameter [m]
        Te       : emission temperature [K]
        qe       : emission specific humidity [kg kg-1]
        Me       : emission mass flux (kg s-1)
    """
    q_th = site['p'] / site['eta']
    lut  = lookup_stacks(site['fuel'], q_th)

    h = site['h'] if site['h'] is not None else lut['h']

    T, w, V_std = lut['T'], lut['w'], lut['V_std']

    # Outlet geometry needs the volume flux at emission conditions.
    V_stack = V_std * T / cst.T0
    A = V_stack / w
    Me = rho_std * V_std

    src = {
        'lat': site['lat'],
        'lon': site['lon'],
        'h': h,
        'd': np.sqrt(4.0 * A / np.pi),
        'Te': T,
        'qe': site['rh'] * qsat(p_env, T),
        'Me': Me,
        }

    return src


if __name__ == '__main__':

    sites = {
        'NA_D': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99353, 'lon': 6.66737},
        'NA_E': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99451, 'lon': 6.66645},
        'NA_F': {'fuel': 'lignite_cooling_tower', 'p':  300, 'eta': 0.31, 'rh': 0.95, 'h': 106, 'lat': 50.99400, 'lon': 6.66878},
        'NA_G': {'fuel': 'lignite_cooling_tower', 'p':  600, 'eta': 0.31, 'rh': 0.95, 'h': 128, 'lat': 50.99514, 'lon': 6.66810},
        'NA_H': {'fuel': 'lignite_cooling_tower', 'p':  600, 'eta': 0.31, 'rh': 0.95, 'h': 128, 'lat': 50.99465, 'lon': 6.67041},
        'NA_K': {'fuel': 'lignite_cooling_tower', 'p': 1027, 'eta': 0.43, 'rh': 0.95, 'h': 200, 'lat': 50.99611, 'lon': 6.67137},
        }

    p_env = 1e5

    for name, site in sites.items():
        print(name, get_source(site, p_env))
