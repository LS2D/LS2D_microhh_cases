import numpy as np

case_name = 'valthermond_ideal'
case = 'convective'
work_dir = 'test_case_{}'.format(case)
type_float = np.float32

xsize = 25600
ysize = 6400
zsize = 3200

itot = 512
jtot = 128
ktot = 128

dust_list = ['0-2um', '2-10um', '10-20um', '20-58um', '58-83um', '83-440um']
dust_diameter = np.array([1, 6, 15, 39, 70.5, 261.5]) * 1e-6

# Calculate gravitational settling velocity dust.
# Following: https://sci-hub.se/10.1063/1.5022089
# NOTE: compared to the paper, this uses a fixed atmospheric density!
rho_p = 1500   # Density particles [kg m-3]
rho_a = 1.225  # Reference density air [kg m-3]
nu = 1e-5      # Kinematic viscosity air [m2 s-1]
g = 9.81       # Gravitational acceleration [m s-2]

tau_p = dust_diameter**2 * rho_p / (18 * nu * rho_a)
w_s = -tau_p * g

"""
Create MicroHH input.
"""
# Simple equidistant vertical grid.
zsize = 3200
ktot = 128
dz = zsize / ktot
z = np.arange(dz/2, zsize, dz)

# Vertical profiles.
zi = 1000
th0 = 290
dth = 10
dthdz = 0.006

th = np.zeros(ktot)
ml = z<zi
th[ml]  = th0
th[~ml] = th0 + dth + (z[~ml] - zi) * dthdz

u = np.ones(ktot) * 5

# Create MicroHH `case_input.nc` input.
init_profiles = {
        'z': z,
        'th': th,
        'u': u}

mlt.write_netcdf_input(
        case_name, 'f8',
        init_profiles = init_profiles)

"""
# Generate .ini file
"""
ini = mlt.read_namelist('{}.ini.base'.format(case_name))

ini['grid']['itot'] = itot
ini['grid']['jtot'] = jtot
ini['grid']['ktot'] = ktot

ini['grid']['xsize'] = xsize
ini['grid']['ysize'] = ysize
ini['grid']['zsize'] = zsize

ini['buffer']['zstart'] = 0.75 * zsize

if case == 'neutral':
    ini['boundary']['sbot[th]'] = 0.
elif case == 'convective':
    ini['boundary']['sbot[th]'] = 0.1
else:
    raise Exception('Unknown case...')

ini['cross']['xz'] = ysize/2
ini['cross']['yz'] = list(np.array([0.25, 0.5, 0.75]) * xsize)
ini['cross']['xy'] = [10, 50, 100, 250, 500]

# Gravitational settling dust.
ini['fields']['slist'] = dust_list
ini['dust']['dustlist'] = dust_list
ini['boundary']['sbot_2d_list'] = dust_list
ini['boundary']['scalar_outflow'] = dust_list
ini['limiter']['limitlist'] = dust_list
ini['advec']['fluxlimit_list'] = dust_list
ini['cross']['crosslist'] += dust_list
for specie in dust_list:
    ini['cross']['crosslist'] += ['{}path'.format(specie)]

# Set settling velocity per dust category:
for i in range(len(dust_list)):
    ini['dust']['ws[{}]'.format(dust_list[i])] = w_s[i]

# Write new .ini file
mlt.write_namelist('{}.ini'.format(case_name), ini)

"""
Dust emission from field.
"""
dx = xsize / itot
dy = ysize / jtot

x = np.arange(dx/2, xsize, dx)
y = np.arange(dy/2, ysize, dy)

field_mask = np.zeros((jtot, itot), dtype=bool)

# Circular emission:
x0 = 2000
y0 = ysize / 2
r = 1000

for j in range(jtot):
    for i in range(itot):
        d = np.sqrt((x[i] - x0)**2  + (y[j]-y0)**2)
        if d < r:
            field_mask[j,i] = True

field_flux = np.zeros((jtot, itot), dtype=type_float)
field_flux[field_mask] = 1.
for scalar in dust_list:
    field_flux.tofile('{}_bot_in.0000000'.format(scalar))

"""
Move files to work directory.
"""
to_move = ['valthermond_ideal.ini', 'valthermond_ideal_input.nc'] + glob.glob('*.0000000')

for f in to_move:
    shutil.move(f, '{}/{}'.format(work_dir, f))
