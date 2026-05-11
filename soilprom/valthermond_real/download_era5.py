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

# Python modules
from datetime import datetime

# Custom modules
import ls2d

# LS2D and custom modules
from settings import ls2d_settings, env

ls2d_settings['start_date'] = datetime(year=2022, month=5, day=10, hour=21)
ls2d_settings['end_date']   = datetime(year=2022, month=5, day=12, hour=0)

# Download ERA5 data
ls2d.download_era5(ls2d_settings)