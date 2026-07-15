import numpy as np

"""
Stack parameters for large combustion plants, digitized from
Pregger & Friedrich (2009), Env. Pollution 157, 552-560, Fig. 1.

Bins are thermal capacity (min_MWth, max_MWth); max = None is open-ended.
  h     : stack height                        [m]
  T     : flue gas temperature                [K]
  w     : flue gas exit velocity              [m s-1], at stack conditions
  V_std : flue gas flow rate, standard state  [m3 s-1], per Eq. (1) R0
"""

T0 = 273.15       # standard-state reference temperature (K)
rho_ref = 1.29    # flue gas density at standard state (kg m-3)

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


def get_stack_properties(fuel, capacity):
    """
    Stack and flue gas properties for a given fuel type and thermal capacity.

    Arguments:
        fuel     : key of `stack_properties`
        capacity : thermal capacity [MW_th]

    Returns dict with the tabulated values plus, derived consistently:
        V_stack : volume flux at stack conditions  [m3 s-1]
        A       : stack outlet area                [m2]
        d       : stack outlet diameter            [m]
        m_dot   : flue gas mass flux               [kg s-1]
    """
    if fuel not in stack_properties:
        raise KeyError(f'unknown fuel `{fuel}`, options: {list(stack_properties)}')

    for entry in stack_properties[fuel]:
        lo, hi = entry['range']
        if capacity >= lo and (hi is None or capacity < hi):
            p = {k: v for k, v in entry.items() if k != 'range'}

            p['V_stack'] = p['V_std'] * p['T'] / T0
            p['A']       = p['V_stack'] / p['w']
            p['d']       = np.sqrt(4.0 * p['A'] / np.pi)
            p['m_dot']   = rho_ref * p['V_std']

            return p

    raise ValueError(f'capacity {capacity} MWth outside tabulated range for `{fuel}`')
