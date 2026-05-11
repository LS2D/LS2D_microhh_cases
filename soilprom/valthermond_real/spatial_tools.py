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

from numba import jit
import numpy as np
import pyproj


@jit(nopython=True, nogil=True)
def distance(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


@jit(nopython=True, nogil=True)
def point_is_on_line(x, y, x1, y1, x2, y2):

    d1 = distance(x,  y,  x1, y1)
    d2 = distance(x,  y,  x2, y2)
    d3 = distance(x1, y1, x2, y2)

    eps = 1e-12
    return np.abs((d1+d2)-d3) < eps


@jit(nopython=True, nogil=True)
def is_left(xp, yp, x0, y0, x1, y1):
    """
    Check whether point (xp,yp) is left of line segment ((x0,y0) to (x1,y1))
    returns:  >0 if left of line, 0 if on line, <0 if right of line
    """

    return (x1-x0) * (yp-y0) - (xp-x0) * (y1-y0)


@jit(nopython=True, nogil=True)
def is_inside(xp, yp, x_set, y_set, size):
    """
    Given location (xp,yp) and set of line segments (x_set, y_set), determine
    whether (xp,yp) is inside (or on) polygon.
    """

    # First simple check on bounds
    if (xp < x_set.min() or xp > x_set.max() or yp < y_set.min() or yp > y_set.max()):
        return False

    wn = 0
    for i in range(size-1):

        # Second check: see if point exactly on line segment:
        if point_is_on_line(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]):
            return False

        #if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) == 0):
        #    return False

        # Calculate winding number
        if (y_set[i] <= yp):
            if (y_set[i+1] > yp):
                if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) > 0):
                    wn += 1
        else:
            if (y_set[i+1] <= yp):
                if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) < 0):
                    wn -= 1

    if (wn == 0):
        return False
    else:
        return True