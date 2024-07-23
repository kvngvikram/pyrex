import numpy as np
from numba import jit
from numba import prange
from config_cross import setup


# # uncomment this to disable numba jit
# def jit(*args, **kwargs):
#     def inner(func):
#         def wrapper(*wargs, **wkwargs):
#             # print('inside wrapper')
#             return func(*wargs, **wkwargs)
#         return wrapper
#     return inner


# Search for Edge_case to find the occurances in this file

# TODO: IMPORTANT: What happens when there are NaNs in lon lat during detection
# So check for NaN_handling search string in comments. Any filteration of NaNs
# in lon lat should not be done without storing the index of the track point
# because in the detection file the indices should match to rows in track files
# But there should be a condition. Clean your data beforehand. lon lats should
# not have NaNs !!

# TODO: there are two versions of get_break_indices. The new one will have
# python interpreter induced optimisations and has a for loop !!!
# So the new one may be used only when numba is enabled. But for finding the
# timing properties old one will be better because no unknown optimisations.
# Whereas the algorithem of new one can have improvement in overall speed of
# detection script during edge cases (only during edge cases).
# So using old one for now.

# TODO: numba_diff_bug: np.diff() when doing for a 1d float array will give error
# if the array is a non-contiguous array. This has someting to do with how
# array is stored internally in RAM.
# A workaround is to use np.diff() on the copy of the array instead
# In the current code some of such scenearios were avoided by using lon_shift()
# or redifining the same array like x = x[flag_array] or even x = x+0
# Search for numba_diff_bug to find the occurances

# TODO: Numba doesn't support cache=True for recursive functions.
# see github.com/numba/numba/issues/6061

# Also general numba bugs can be searched by numba_bug

semi_major_axis = setup().semi_major_axis
flattening_coefficient = setup().flattening_coefficient


lon_shift_condition_value = (setup().lon_shift_condition_value
                             if hasattr(setup, "lon_shift_condition_value")
                             else 180)

recurse_level_limit = (setup().recurse_level_limit
                       if hasattr(setup, "recurse_level_limit")
                       else 50)


# doing this because numba doesn't support arguments for max and min functions
@jit(nopython=True, parallel=True, cache=True)
def min_ax1(xa, xb):
    x_min = np.zeros(xa.size)
    for i in prange(xa.size):
        x_min[i] = xa[i] if xa[i] < xb[i] else xb[i]
    return x_min


@jit(nopython=True, parallel=True, cache=True)
def max_ax1(xa, xb):
    x_max = np.zeros(xa.size)
    for i in prange(xa.size):
        x_max[i] = xa[i] if xa[i] > xb[i] else xb[i]
    return x_max


@jit(nopython=True, parallel=True, cache=True)
def sort_ax1_4(x1a, x1b, x2a, x2b):
    x_sort = np.zeros(x1a.size*4).reshape(x1a.size, 4)
    array2d = np.column_stack((x1a, x1b, x2a, x2b))
    for i in prange(x1a.size):
        x_sort[i] = np.sort(array2d[i])
    return x_sort


@jit(nopython=True, parallel=True, cache=True)
def mean_ax1(x):
    x_mean = np.zeros(x.shape[0])
    for i in prange(x_mean.size):
        x_mean[i] = np.mean(x[i])
    return x_mean


# TODO: using this until numba has an official np.isin() support
@jit(nopython=True, parallel=True, cache=True)
def my_isin_1d(a, b):
    out_f = np.zeros(a.size) > 1  # dtype aurguments doesn't work
    for i in prange(out_f.size):
        out_f[i] = True if (b == a[i]).sum() > 0 else False
    return out_f


# to shift longitude range from 0 <-> 360 to -180 <-> 180
@jit(nopython=True, cache=True)
def lon_shift(lon):
    return lon - 360*(lon > 180)


@jit(nopython=True, cache=True, parallel=True)
def np_cosd(angle_degrees):
    return np.cos(angle_degrees * np.pi/180)


@jit(nopython=True, cache=True, parallel=True)
def np_sind(angle_degrees):
    return np.sin(angle_degrees * np.pi/180)


@jit(nopython=True, cache=True)
def get_pseudo_segment_flags(x, y):
    """
    Give a boolean array of size 1 less then (x, y) track points. Each flag is
    for the segments inbetween the points.
    Give True if the length of segment is 0 i.e. both points are exactly same.
    If no occurances are present giving a numpy bool array with just one False
    This is done to decreace the efforts when used with cases with all False
    """
    x = np.copy(x)  # for explanation search for numba_diff_bug in this file
    y = np.copy(y)
    ps_f = np.logical_and((np.diff(x) == 0), (np.diff(y) == 0))
    # TODO this is done for improving efficiency when used at
    # get_recursive_box_indices()
    # ps_f.sum() will be faster when there is just one False rather than many
    # elements with just False but giving same outcome
    if ps_f.sum() > 0:  # all are False
        return ps_f
    else:
        return np.array([False])


# @jit(nopython=True)
# @jit(nopython=True, parallel=True, cache=True)
@jit(nopython=True, cache=True)
def haversine(lon1_deg, lat1_deg, lon2_deg, lat2_deg):
    """
    Calculate the great circle distance between two points
    on the planet (specified in decimal degrees)

    from https://stackoverflow.com/a/15737218
    refer https://en.wikipedia.org/wiki/Great-circle_distance
    """
    # convert decimal degrees to radians
    lon1 = lon1_deg * np.pi/180
    lat1 = lat1_deg * np.pi/180
    lon2 = lon2_deg * np.pi/180
    lat2 = lat2_deg * np.pi/180

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))  # np.sqrt gives non-negative value

    local_planet_radius_in_m = planet_radius_simple_ellipsoid(
            (lat1_deg + lat2_deg)/2)

    return local_planet_radius_in_m * c


# @jit(nopython=True, parallel=True, cache=True)
@jit(nopython=True, cache=True)
def planet_radius_simple_ellipsoid(lat_deg):
    a = semi_major_axis
    f = flattening_coefficient
    b = a*(1 - f)  # semi minor axis from semi major axis and flattening coeff

    lat_rad = lat_deg * np.pi/180  # converting into radians
    r = np.sqrt((a*np.cos(lat_rad))**2 + (b*np.sin(lat_rad))**2)

    return r


@jit(nopython=True, parallel=True, cache=True)
def old_get_break_indices(x, y):  # given like this for convenience with P.map
    # n = x.size

    x = lon_shift(x)  # just in case if its not done beforehand

    x = np.copy(x)  # For explanation search for numba_diff_bug in this file
    y = np.copy(y)

    # for +180-180 crossing
    break_i = np.where(np.abs(np.diff(x)) > lon_shift_condition_value)[0]
    # for trend changing
    tmp_x = np.where(np.abs(np.diff(np.sign(np.diff(x)))))[0] + 1
    tmp_y = np.where(np.abs(np.diff(np.sign(np.diff(y)))))[0] + 1

    break_i = np.concatenate((break_i, tmp_x, tmp_y))

    break_i = np.unique(break_i)

    # there is already an additional 0 in the beginning during initialisation
    break_i = np.concatenate((np.array([0]), break_i, np.array([x.size])))
    # for usage as
    # for i in range(break_i.size - 1):
    #     xp = x[break_i[i]: break_i[i+1] + 1]
    #     # use xp
    # this will xp is a part of x and for loop over i will cover entire xp
    # in parts

    return break_i


@jit(nopython=True, cache=True)
def get_break_indices(x, y):
    """
    for usage as:
    >    break_i = this_function(x, y)
    >    for i in range(break_i.size - 1):
    >        xp = x[break_i[i]: break_i[i+1] + 1]
    >        yp = y[break_i[i]: break_i[i+1] + 1]
    >        # use xp and yp

    (xp, yp) together is a subtrack and by looping over break indices the
    entire track will be covered in multiple parts.

    Things like having a zero as the first element of return array, subracting
    1 from all elements and adding the last element done to make the return
    work like in the above for loop and get the sub-tracks/parts correctly.

    Also the edge cases where lon or lat doesn't change i.e. same exact value
    because perfect horizontal or vertical track will not cause additional
    breaks

    Exclusive NaN_handling section was mitigated. It will be done further in
    get_recursive_box_indices()
    """
    x = lon_shift(x)  # just in case if its not done beforehand

    # doing for longitude first, which also includes 180 crossing condition
    before_diff = x[1] - x[0]

    # checking the 180 crossing for first segment. Also making the first
    # element as 1 here because at the end -1 will be done for all the elements
    # of break_i array once. Finally the first element of break_i must be 0
    if before_diff > lon_shift_condition_value:
        break_i = np.array([1, 1])
    else:
        break_i = np.array([1])

    for i in np.arange(x.size-2) + 2:
        new_diff = x[i] - x[i-1]
        if ((new_diff * before_diff < 0) or
           abs(new_diff) > lon_shift_condition_value):
            break_i = np.append(break_i, i)

        # for edge case when values are not changing i.e. new_diff = 0
        before_diff = new_diff if abs(new_diff) > 0 else before_diff
        # before_diff = new_diff if new_diff != 0 else before_diff
        # when the consecutive elements of x are exactly same i.e. new_diff is
        # zero but at the same time if its also a point of maxima/minima
        # the change in trend on both sides of maxima/minima will not be
        # caught if diff is taken as 0 and before_diff*new_diff will be 0 in
        # next loop and the if condition will not be flagged
        # Basically don't overwrite/forget the before_diff diff if new_diff is
        # bad. Works when new diff is NaN as well i.e. when there are NaNs in
        # x, y (NaN_handling)

    # same for y
    before_diff = y[1] - y[0]
    for i in np.arange(y.size-2) + 2:
        new_diff = y[i] - y[i-1]
        if new_diff * before_diff < 0:
            break_i = np.append(break_i, [i])
        before_diff = new_diff if new_diff != 0 else before_diff

    # there may be common trend changes in both x and y
    break_i = np.unique(break_i)

    break_i = break_i - 1
    break_i = np.append(break_i, np.array([x.size-1]))

    return break_i


# this is for small distances using haversine function
@jit(nopython=True, parallel=True, cache=True)
def distance_on_track(lon, lat):
    """
    Distance along the track based on haversine formula
    """

    nan_flag = np.isnan(lon) | np.isnan(lat)
    size = nan_flag.size

    lon = lon[~nan_flag]
    lat = lat[~nan_flag]

    lona = lon[:-1]
    lonb = lon[1:]
    lata = lat[:-1]
    latb = lat[1:]

    distance_array = haversine(lona, lata, lonb, latb)

    # we have filtered out some nan values
    #
    # IF
    # lon                  : nan   1.1   2   nan   3    3    ->  size
    # lat                  : nan   1.1   2   nan   3    3    ->  size
    # THEN
    # nan_flag             :  T     F    F    T    F    F    ->  size
    #                                 \/   \_____/   \/
    # distance_array       :          d1        d2   d3      ->  nan_flag.sum()
    #                          \___/ \__/ \__/ \__/ \__/
    # spaced_distance_array:     0    d1   d2    0   d3      ->  size -1
    #
    # cumsum               :     0    d1   d1+  d1+  d1+
    #                                      d2   d2   d2+     ->  size -1
    #                                                d3
    spaced_distance_array = np.zeros(size - 1)
    i_da = 0  # index for distance array variable
    for i in range(size - 1):
        if not nan_flag[i]:
            spaced_distance_array[i] = distance_array[i_da]
            i_da += 1

    cumsum = spaced_distance_array.cumsum()  # cumulative sum
    cumsum[nan_flag[1:]] = np.nan

    # return cumulative sum of same size as input
    cumsum_start = 0.0 if not nan_flag[0] else np.nan
    # return same size as the input track
    return np.concatenate((np.array([cumsum_start]), cumsum))


# TODO
# recrusion with cache seems to have a problem
# see github.com/numba/numba/issues/6061
# @jit(nopython=True, cache=True)
@jit(nopython=True)
def recurse_by_2(x1p, y1p, i1p, x2p, y2p, i2p, ls1_idx, ls2_idx,
                 recurse_level_limit,
                 level=1):
    """
    """
    # print(level)
    # remove NaNs before hand ! no need to worry about ordering or skipping as
    # index values of correspoinding x, y are always there

    min_x1 = min(x1p[0], x1p[-1])
    max_x1 = max(x1p[0], x1p[-1])
    min_y1 = min(y1p[0], y1p[-1])
    max_y1 = max(y1p[0], y1p[-1])
    min_x2 = min(x2p[0], x2p[-1])
    max_x2 = max(x2p[0], x2p[-1])
    min_y2 = min(y2p[0], y2p[-1])
    max_y2 = max(y2p[0], y2p[-1])

    big_box_flag = ((max_x1 >= min_x2) &
                    (min_x1 <= max_x2) &
                    (max_y1 >= min_y2) &
                    (min_y1 <= max_y2))

    if big_box_flag:
        n1 = i1p.size
        n2 = i2p.size

        if n1 > 2 and n2 > 2 and level <= recurse_level_limit:

            # dividing each track into part 1 and part 2 i.e. _p1 and _p2
            x1_p1 = x1p[0:int(n1/2)+1]
            y1_p1 = y1p[0:int(n1/2)+1]
            i1_p1 = i1p[0:int(n1/2)+1]

            x2_p1 = x2p[0:int(n2/2)+1]
            y2_p1 = y2p[0:int(n2/2)+1]
            i2_p1 = i2p[0:int(n2/2)+1]

            x1_p2 = x1p[int(n1/2):]
            y1_p2 = y1p[int(n1/2):]
            i1_p2 = i1p[int(n1/2):]

            x2_p2 = x2p[int(n2/2):]
            y2_p2 = y2p[int(n2/2):]
            i2_p2 = i2p[int(n2/2):]

            # part 1 of first track
            ls1_idx, ls2_idx = recurse_by_2(x1_p1, y1_p1, i1_p1,
                                            x2_p1, y2_p1, i2_p1,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

            ls1_idx, ls2_idx = recurse_by_2(x1_p1, y1_p1, i1_p1,
                                            x2_p2, y2_p2, i2_p2,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

            # part 2 of first track
            ls1_idx, ls2_idx = recurse_by_2(x1_p2, y1_p2, i1_p2,
                                            x2_p1, y2_p1, i2_p1,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

            ls1_idx, ls2_idx = recurse_by_2(x1_p2, y1_p2, i1_p2,
                                            x2_p2, y2_p2, i2_p2,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

        elif n2 > 2 and level <= recurse_level_limit:

            x2_p1 = x2p[0:int(n2/2)+1]
            y2_p1 = y2p[0:int(n2/2)+1]
            i2_p1 = i2p[0:int(n2/2)+1]

            x2_p2 = x2p[int(n2/2):]
            y2_p2 = y2p[int(n2/2):]
            i2_p2 = i2p[int(n2/2):]

            ls1_idx, ls2_idx = recurse_by_2(x1p, y1p, i1p,
                                            x2_p1, y2_p1, i2_p1,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

            ls1_idx, ls2_idx = recurse_by_2(x1p, y1p, i1p,
                                            x2_p2, y2_p2, i2_p2,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

        elif n1 > 2 and level <= recurse_level_limit:

            x1_p1 = x1p[0:int(n1/2)+1]
            y1_p1 = y1p[0:int(n1/2)+1]
            i1_p1 = i1p[0:int(n1/2)+1]

            x1_p2 = x1p[int(n1/2):]
            y1_p2 = y1p[int(n1/2):]
            i1_p2 = i1p[int(n1/2):]

            ls1_idx, ls2_idx = recurse_by_2(x1_p1, y1_p1, i1_p1,
                                            x2p, y2p, i2p,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

            ls1_idx, ls2_idx = recurse_by_2(x1_p2, y1_p2, i1_p2,
                                            x2p, y2p, i2p,
                                            ls1_idx, ls2_idx,
                                            recurse_level_limit,
                                            level=level+1)

        else:
            # now mostly both tracks are only one line segment, but if the
            # recurse_level_limit is reached then they can be larger,
            # so useing a for loop
            for i in range(n1-1):
                for j in range(n2-1):
                    x1a = x1p[:-1][i]
                    y1a = y1p[:-1][i]
                    i1a = i1p[:-1][i]

                    x2a = x2p[:-1][j]
                    y2a = y2p[:-1][j]
                    i2a = i2p[:-1][j]

                    x1b = x1p[1:][i]
                    y1b = y1p[1:][i]
                    x2b = x2p[1:][j]
                    y2b = y2p[1:][j]

                    x1_max = max(x1a, x1b)
                    y1_max = max(y1a, y1b)
                    x1_min = min(x1a, x1b)
                    y1_min = min(y1a, y1b)
                    x2_max = max(x2a, x2b)
                    y2_max = max(y2a, y2b)
                    x2_min = min(x2a, x2b)
                    y2_min = min(y2a, y2b)

                    # the rectangle test
                    box_flag = ((x1_max >= x2_min) &
                                (x1_min <= x2_max) &
                                (y1_max >= y2_min) &
                                (y1_min <= y2_max))

                    if box_flag:
                        ls1_idx = np.append(ls1_idx, i1a)
                        ls2_idx = np.append(ls2_idx, i2a)

    return ls1_idx, ls2_idx


# TODO
# recrusion with cache seems to have a problem
# see github.com/numba/numba/issues/6061
# @jit(nopython=True, cache=True)
@jit(nopython=True)
def recurse_by_n(x1, y1, i1, x2, y2, i2, ls1_idx, ls2_idx,
                 rn, recurse_level_limit,
                 level=1):
    """
    x1, y1 are longitude latitude of first track
    i1 are indices assigned beforehand so that we keep track of data while
    splitting/breaking tracks.
    x1, y1, and i1 sizes will be arrays same size of track 1
    x2, y2, and i2 sizes will be arrays same size of track 2

    ls1_idx, ls2_idx initialised with a single element array before calling the
    function and at the end indices of the flagged segments will be appended.
    These are the output of the function.

    rn is the recursion type, basically recurse by splitting the track into
    ls2_idx parts or 3 parts of even 10 parts

    recurse_level_limit the limit for the recursion depth, just a functionality
    , but I generally give a large number like 50 and don't touch it

    level to keep track of the present level, used it for debugging
    """
    # print(level)
    # remove NaNs before hand ! no need to worry about ordering or skipping as
    # index values of correspoinding x, y are always there

    min_x1 = min(x1[0], x1[-1])
    max_x1 = max(x1[0], x1[-1])
    min_y1 = min(y1[0], y1[-1])
    max_y1 = max(y1[0], y1[-1])
    min_x2 = min(x2[0], x2[-1])
    max_x2 = max(x2[0], x2[-1])
    min_y2 = min(y2[0], y2[-1])
    max_y2 = max(y2[0], y2[-1])

    big_box_flag = ((max_x1 >= min_x2) &  # checking the box overlap
                    (min_x1 <= max_x2) &
                    (max_y1 >= min_y2) &
                    (min_y1 <= max_y2))

    if not big_box_flag:  # EXIT the function !!
        return ls1_idx, ls2_idx

    n1 = i1.size
    n2 = i2.size

    if n1 > rn and n2 > rn and level <= recurse_level_limit:

        # the following are just some numbers used to get the right slicing
        # for splitting the tracks
        b1 = (n1-1)//rn
        b2 = (n2-1)//rn

        # check all the combinations of the splits of tracks
        for i in range(rn):
            for j in range(rn):

                s1 = b1*i
                e1 = b1*(i+1)+1 if i < (rn-1) else n1
                s2 = b2*j
                e2 = b2*(j+1)+1 if j < (rn-1) else n2

                x1p = x1[s1:e1]  # part one
                y1p = y1[s1:e1]
                i1p = i1[s1:e1]
                x2p = x2[s2:e2]  # part two
                y2p = y2[s2:e2]
                i2p = i2[s2:e2]

                # if i < (rn-1):  # alternate way of splitting
                #     x1p = x1[b1*i:b1*(i+1)+1]
                #     y1p = y1[b1*i:b1*(i+1)+1]
                #     i1p = i1[b1*i:b1*(i+1)+1]
                # else:
                #     x1p = x1[b1*i:]
                #     y1p = y1[b1*i:]
                #     i1p = i1[b1*i:]

                # if j < (rn-1):
                #     x2p = x2[b2*j:b2*(j+1)+1]
                #     y2p = y2[b2*j:b2*(j+1)+1]
                #     i2p = i2[b2*j:b2*(j+1)+1]
                # else:
                #     x2p = x2[b2*j:]
                #     y2p = y2[b2*j:]
                #     i2p = i2[b2*j:]

                ls1_idx, ls2_idx = recurse_by_n(x1p, y1p, i1p,  # recursing
                                                x2p, y2p, i2p,
                                                ls1_idx, ls2_idx,
                                                rn, recurse_level_limit,
                                                level=level+1)

    # what if one track has reached segment level but not the other one?
    elif n2 > rn and level <= recurse_level_limit:

        b2 = (n2-1)//rn

        for j in range(rn):

            s2 = b2*j
            e2 = b2*(j+1)+1 if j < (rn-1) else n2

            x2p = x2[s2:e2]
            y2p = y2[s2:e2]
            i2p = i2[s2:e2]

            ls1_idx, ls2_idx = recurse_by_n(x1, y1, i1,
                                            x2p, y2p, i2p,
                                            ls1_idx, ls2_idx,
                                            rn, recurse_level_limit,
                                            level=level+1)

    elif n1 > rn and level <= recurse_level_limit:

        b1 = (n1-1)//rn

        for i in range(rn):

            s1 = b1*i
            e1 = b1*(i+1)+1 if i < (rn-1) else n1

            x1p = x1[s1:e1]
            y1p = y1[s1:e1]
            i1p = i1[s1:e1]

            ls1_idx, ls2_idx = recurse_by_n(x1p, y1p, i1p,
                                            x2, y2, i2,
                                            ls1_idx, ls2_idx,
                                            rn, recurse_level_limit,
                                            level=level+1)

    else:
        # now mostly both tracks are only one line segment, but if the
        # recurse_level_limit is reached then they can be larger,
        # so useing a for loop
        # print(f"bruting {n1} {n2}")
        for i in range(n1-1):
            for j in range(n2-1):
                x1a = x1[:-1][i]
                y1a = y1[:-1][i]
                i1a = i1[:-1][i]

                x2a = x2[:-1][j]
                y2a = y2[:-1][j]
                i2a = i2[:-1][j]

                x1b = x1[1:][i]
                y1b = y1[1:][i]
                x2b = x2[1:][j]
                y2b = y2[1:][j]

                x1_max = max(x1a, x1b)
                y1_max = max(y1a, y1b)
                x1_min = min(x1a, x1b)
                y1_min = min(y1a, y1b)
                x2_max = max(x2a, x2b)
                y2_max = max(y2a, y2b)
                x2_min = min(x2a, x2b)
                y2_min = min(y2a, y2b)

                # the rectangle test
                box_flag = ((x1_max >= x2_min) &
                            (x1_min <= x2_max) &
                            (y1_max >= y2_min) &
                            (y1_min <= y2_max))

                if box_flag:
                    ls1_idx = np.append(ls1_idx, i1a)
                    ls2_idx = np.append(ls2_idx, i2a)

    return ls1_idx, ls2_idx


@jit(nopython=True)  # TODO: BUG: do not use parallel here
def get_recursive_box_indices(x1, y1, x2, y2,
                              bi_1=None, bi_2=None,
                              ps_f_1=None, ps_f_2=None,
                              internal_flag=True,
                              min_x1=-np.inf, max_x1=np.inf,
                              min_y1=-np.inf, max_y1=np.inf,
                              min_x2=-np.inf, max_x2=np.inf,
                              min_y2=-np.inf, max_y2=np.inf,
                              recurse_level_limit=recurse_level_limit,
                              rn=None):

    # If there is no overall overlap in the tracks then there is no point of
    # breaking them further. So just return empty arrays.
    # Also if the default values of min max are taken as infinity because why
    # calculate them again using np.min() np.max(). These may not be as fast
    # as taking first and last elements as min max post breaking at trend
    # changes
    big_box_flag = ((max_x1 >= min_x2) &
                    (min_x1 <= max_x2) &
                    (max_y1 >= min_y2) &
                    (min_y1 <= max_y2))
    if not big_box_flag:
        ls1_idx = np.array([-2])  # just in case numba complains about dtype
        ls2_idx = np.array([-2])
        return ls1_idx[1:], ls2_idx[2:]

    bi_1 = get_break_indices(x1, y1) if bi_1 is None else bi_1
    bi_2 = get_break_indices(x2, y2) if bi_2 is None else bi_2

    ps_f_1 = get_pseudo_segment_flags(x1, y1) if ps_f_1 is None else ps_f_1
    ps_f_2 = get_pseudo_segment_flags(x2, y2) if ps_f_2 is None else ps_f_2

    n1, n2 = x1.size, x2.size
    i1 = np.arange(n1)
    i2 = np.arange(n2)
    ls1_idx = np.array([-2])
    ls2_idx = np.array([-2])
    for i in range(bi_1.size - 1):
        for j in range(bi_2.size - 1):

            if internal_flag and (i == j):
                # since bi_ , x, y, i1/i2 will be exactly same for internal
                # check of same track
                # and when i and j are same here, there will not be any cross
                # but all the points will be flagged (in recurse func)
                continue  # so skip the i, j pair

            x1p = x1[bi_1[i]: bi_1[i+1] + 1]
            y1p = y1[bi_1[i]: bi_1[i+1] + 1]
            i1p = i1[bi_1[i]: bi_1[i+1] + 1]

            x2p = x2[bi_2[j]: bi_2[j+1] + 1]
            y2p = y2[bi_2[j]: bi_2[j+1] + 1]
            i2p = i2[bi_2[j]: bi_2[j+1] + 1]

            # # NaN_handling
            # # it should be here, after the i1 and i2 are defined and after
            # # splitting based on trends. Because NaN values can be filtered out
            # # only after defining i1, i2. And this filtering should not be
            # # done after calculating trend change break indices. And since
            # # NaN filteration is done only in module functions, this is the
            # # only place
            # # This filtering should be done because recurse function may
            # # arbitarily split these and when taking end points as min max
            # # it may take NaN as max and then do box overlap evaluation
            # nan_flag = np.logical_or(np.isnan(x1p), np.isnan(y1p))
            # x1p = x1p[~nan_flag]
            # y1p = y1p[~nan_flag]
            # i1p = i1p[~nan_flag]
            # nan_flag = np.logical_or(np.isnan(x2p), np.isnan(y2p))
            # x2p = x2p[~nan_flag]
            # y2p = y2p[~nan_flag]
            # i2p = i2p[~nan_flag]
            # # Note: break indices will not give just one point and one nan
            # # so these sub-tracks/parts will never be just one point

            # if a part is at longitude crossing then do the calculation in
            # 0-360 instead
            if x1p.size == 2:
                if abs(x1p[0] - x1p[-1]) > lon_shift_condition_value:
                    x1p = x1p + 360*(x1p < 0)
            if x2p.size == 2:
                if abs(x2p[0] - x2p[-1]) > lon_shift_condition_value:
                    x2p = x2p + 360*(x2p < 0)

            if rn is None:

                ls1_idx, ls2_idx = recurse_by_2(x1p, y1p, i1p,
                                                x2p, y2p, i2p,
                                                ls1_idx, ls2_idx,
                                                recurse_level_limit)
            else:

                ls1_idx, ls2_idx = recurse_by_n(x1p, y1p, i1p,
                                                x2p, y2p, i2p,
                                                ls1_idx, ls2_idx,
                                                rn, recurse_level_limit)
    ls1_idx = ls1_idx[1:]  # since we had a -2 in the beginning
    ls2_idx = ls2_idx[1:]

    # Edge_case
    # What if there are Pseudo segments, i.e. at the exact same location
    # multiple measurements were taken?
    if (ps_f_1.sum() > 0) or (ps_f_2.sum() > 0):
        ps_idx_f1 = my_isin_1d(ls1_idx, np.where(ps_f_1)[0])  # same size as ls1_idx  # noqa
        ps_idx_f2 = my_isin_1d(ls2_idx, np.where(ps_f_2)[0])  # and ls2_idx

        ps_idx_f = np.logical_or(ps_idx_f1, ps_idx_f2)

        ls1_idx = ls1_idx[~ps_idx_f]
        ls2_idx = ls2_idx[~ps_idx_f]

    if internal_flag:
        internal_filter = np.abs(ls1_idx - ls2_idx) > 1  # this is logic
        # avoiding consecutive segments as having an intersection
        # remove (ls1_idx, ls2_idx) row when this flag is False

        # filter them out
        ls1_idx = ls1_idx[internal_filter]
        ls2_idx = ls2_idx[internal_filter]

        # Edge_case
        # Pseudo segments are already removed, but now this is for the flagging
        # done between the two segments on the either side of the pseudo
        # segments. This happens for internal crossover and the internal_filter
        # cannot remove it.
        # for edge case. For example, when internal crossover checks:
        # A point is measured by ship, moves, and measures at the second point
        # some 2-3 times and then moves to third point for measurement.
        # Now there are two actual segments and some pseudo segments between
        # the two segments which are also in contack at edge points.
        # Due to the contact at the ends of these two segmetns, they should
        # not be flagged. If there were no pseudo segments then these kinds
        # will be filtered out in the step just before (internal_filter).
        if ps_f_1 is not None:  # ps_f_1 and ps_f_2 are same. Its internal
            no_ps_inbetween = np.zeros(ls1_idx.size)  # initialising
            l_ls_idx = min_ax1(ls1_idx, ls2_idx)  # lower ls idx among the 2
            h_ls_idx = max_ax1(ls1_idx, ls2_idx)  # higher ls idx

            for i in range(ls1_idx.size):
                no_ps_inbetween[i] = ps_f_1[l_ls_idx[i]+1:h_ls_idx[i]].sum()

            ps_filter = np.abs(h_ls_idx - l_ls_idx - no_ps_inbetween) > 1

            # filter them out
            ls1_idx = ls1_idx[ps_filter]
            ls2_idx = ls2_idx[ps_filter]

        # when doing internal crossover detection, each intersectin will be
        # flagged twice because both track1 and track2 are same, so these
        # should be removed
        # Example case:
        # ls1_idx = np.array([ 3,  7, 21, 34, 40, 56])
        # ls2_idx = np.array([40, 21,  7, 56,  3, 34])
        # Required:
        # ls1_idx = np.array([21, 40, 56])
        # ls2_idx = np.array([ 7,  3, 34])
        tmp_n = ls1_idx.size
        i = 0
        while ls1_idx.size > tmp_n/2:
            i1 = ls1_idx[i]
            i2 = ls2_idx[i]
            f1 = True if i1 in ls2_idx[i+1:] else False
            f2 = True if i2 in ls1_idx[i+1:] else False
            if f1 and f2:
                ls1_idx = np.delete(ls1_idx, i)
                ls2_idx = np.delete(ls2_idx, i)
            else:
                i += 1

    return ls1_idx, ls2_idx


# TODO: Edge_case: totally horizontal and totally verticle segments are dealt
# with using angle_accuracy_threshold_deg which is basically an extra margin
# in latitude/longitude degree terms within which an intersectin point can lie.
# 2e-6 corresponts to ~0.55 meters at equator
@jit(nopython=True, cache=True, parallel=True)
def intersection_of_arcs_on_sphere(lon1a, lat1a, lon1b, lat1b,
                                   lon2a, lat2a, lon2b, lat2b,
                                   # angle_accuracy_threshold_rad=1e-7,
                                   angle_accuracy_threshold_deg=2e-6,
                                   ):

    # lon1a = x1a
    # lat1a = y1a
    # lon1b = x1b
    # lat1b = y1b
    # lon2a = x2a
    # lat2a = y2a
    # lon2b = x2b
    # lat2b = y2b

    # # for testing
    # lon1a = np.array([45, 46, 0, 179])
    # lon1b = np.array([48, 45, 1, -179])
    # lat1a = np.array([45, 46, 0, 45])
    # lat1b = np.array([48, 45, 1, 49])
    # lon2a = np.array([45, 45, 3, 179])
    # lon2b = np.array([48, 46, 4, -179])
    # lat2a = np.array([48, 46, 4, 48])
    # lat2b = np.array([45, 45, 3, 45])

    # Note: x, y, z here are in ECEF frame!! until the end of this function
    # refer: https://blog.mbedded.ninja/mathematics/geometry/spherical-geometry/finding-the-intersection-of-two-arcs-that-lie-on-a-sphere/  # noqa
    # for converting from lon lat to xyz notations are different from blog
    # https://gssc.esa.int/navipedia/index.php/Ellipsoidal_andCartesian_Coordinates_Conversion
    # this website is referred assuming sphere and no height
    x1a = np_cosd(lat1a) * np_cosd(lon1a)
    y1a = np_cosd(lat1a) * np_sind(lon1a)
    z1a = np_sind(lat1a)

    x1b = np_cosd(lat1b) * np_cosd(lon1b)
    y1b = np_cosd(lat1b) * np_sind(lon1b)
    z1b = np_sind(lat1b)

    x2a = np_cosd(lat2a) * np_cosd(lon2a)
    y2a = np_cosd(lat2a) * np_sind(lon2a)
    z2a = np_sind(lat2a)

    x2b = np_cosd(lat2b) * np_cosd(lon2b)
    y2b = np_cosd(lat2b) * np_sind(lon2b)
    z2b = np_sind(lat2b)

    # initialising return values
    lon_int = lon1a * np.nan  # with nans
    lat_int = lon1a * np.nan
    valid_int_flag = (lon1a*0 + 1) > 0  # with Trues

    # aatr = angle_accuracy_threshold_rad  # tired of long names
    aatd = angle_accuracy_threshold_deg

    # TODO:numba
    # Have to do this because numba doesn't support more than 2 input arguments
    # in np.cross() function
    for i in prange(x1a.size):

        A1 = np.array([x1a[i], y1a[i], z1a[i]])  # vectors for convinence
        B1 = np.array([x1b[i], y1b[i], z1b[i]])
        A2 = np.array([x2a[i], y2a[i], z2a[i]])
        B2 = np.array([x2b[i], y2b[i], z2b[i]])

        N1 = np.cross(A1, B1)
        N2 = np.cross(A2, B2)
        L = np.cross(N1, N2)

        # There will be two intersection points both antipodal to each other.
        I1 = L/L.dot(L)**0.5  # just normalised L
        I2 = -I1

        # in the blog, selection between these two intersection points is done
        # by looking at angle between I1 or I2 with A1, B1, A2, B2.
        # It is implemented in the later lines but it had issues.
        # It involves measuring angle between two vectors which has errors
        # due to computer. So comparisions between angles was not possible
        # when data points are too close
        # So instead reverting back to the simple line intersection way
        # which i avoided in the first place because it was not "elegant",
        # using lon shift at +180 and all. Did not have to worry about This
        # lon shift if we were dealing with angle between vectors :(
        lat_int1 = np.arcsin(I1[2]) * 180/np.pi
        lon_int1 = np.arctan2(I1[1], I1[0]) * 180/np.pi
        lat_int2 = np.arcsin(I2[2]) * 180/np.pi
        lon_int2 = np.arctan2(I2[1], I2[0]) * 180/np.pi

        # if any of the track segment crosses +180 line then do the entire
        # calculation in 0-360 range
        lon_shift_flag_1 = abs(lon1a[i] - lon1b[i]) > lon_shift_condition_value
        lon_shift_flag_2 = abs(lon2a[i] - lon2b[i]) > lon_shift_condition_value
        lon_shift_flag = lon_shift_flag_1 or lon_shift_flag_2

        if lon_shift_flag:
            lon_int1 = lon_int1 + 360*(lon_int1 < 0)
            lon_int2 = lon_int2 + 360*(lon_int2 < 0)

            # overwriting here !! will not be used later though
            lon1a[i] = lon1a[i] + 360*(lon1a[i] < 0)
            lon1b[i] = lon1b[i] + 360*(lon1b[i] < 0)
            lon2a[i] = lon2a[i] + 360*(lon2a[i] < 0)
            lon2b[i] = lon2b[i] + 360*(lon2b[i] < 0)

        valid_int_for_I1_lon = ((lon_int1 < max(lon1a[i], lon1b[i]) + aatd) &
                                (lon_int1 > min(lon1a[i], lon1b[i]) - aatd) &
                                (lon_int1 < max(lon2a[i], lon2b[i]) + aatd) &
                                (lon_int1 > min(lon2a[i], lon2b[i]) - aatd))

        valid_int_for_I1_lat = ((lat_int1 < max(lat1a[i], lat1b[i]) + aatd) &
                                (lat_int1 > min(lat1a[i], lat1b[i]) - aatd) &
                                (lat_int1 < max(lat2a[i], lat2b[i]) + aatd) &
                                (lat_int1 > min(lat2a[i], lat2b[i]) - aatd))

        valid_int_for_I2_lon = ((lon_int2 < max(lon1a[i], lon1b[i]) + aatd) &
                                (lon_int2 > min(lon1a[i], lon1b[i]) - aatd) &
                                (lon_int2 < max(lon2a[i], lon2b[i]) + aatd) &
                                (lon_int2 > min(lon2a[i], lon2b[i]) - aatd))

        valid_int_for_I2_lat = ((lat_int2 < max(lat1a[i], lat1b[i]) + aatd) &
                                (lat_int2 > min(lat1a[i], lat1b[i]) - aatd) &
                                (lat_int2 < max(lat2a[i], lat2b[i]) + aatd) &
                                (lat_int2 > min(lat2a[i], lat2b[i]) - aatd))

        valid_int_for_I1 = valid_int_for_I1_lon and valid_int_for_I1_lat
        valid_int_for_I2 = valid_int_for_I2_lon and valid_int_for_I2_lat

        if valid_int_for_I1:
            lat_int[i] = lat_int1
            lon_int[i] = lon_int1
        elif valid_int_for_I2:
            lat_int[i] = lat_int2
            lon_int[i] = lon_int2
        else:
            valid_int_flag[i] = False

        # # selection of the intersectin point among the two done using the
        # # angle made between start, end points and the intersection points.
        # # refer blog
        # # A1 B1 and I1
        # theta_A1_I1 = np.arccos(A1.dot(I1)/
        #                         A1.dot(A1)**0.5 * I1.dot(I1)**0.5)
        # theta_B1_I1 = np.arccos(B1.dot(I1)/
        #                         B1.dot(B1)**0.5 * I1.dot(I1)**0.5)
        # theta_A1_B1 = np.arccos(A1.dot(B1)/
        #                         A1.dot(A1)**0.5 * B1.dot(B1)**0.5)

        # I1_between_A1_B1 = abs(theta_A1_B1 - theta_A1_I1 - theta_B1_I1) < aatr  # noqa

        # # A2 B2 and I1
        # theta_A2_I1 = np.arccos(A2.dot(I1)/
        #                         A2.dot(A2)**0.5 * I1.dot(I1)**0.5)
        # theta_B2_I1 = np.arccos(B2.dot(I1)/
        #                         B2.dot(B2)**0.5 * I1.dot(I1)**0.5)
        # theta_A2_B2 = np.arccos(A2.dot(B2)/
        #                         A2.dot(A2)**0.5 * B2.dot(B2)**0.5)

        # I1_between_A2_B2 = abs(theta_A2_B2 - theta_A2_I1 - theta_B2_I1) < aatr  # noqa

        # valid_int_for_I1 = I1_between_A1_B1 and I1_between_A2_B2

        # # A1 B1 I2
        # theta_A1_I2 = np.arccos(A1.dot(I2)/
        #                         A1.dot(A1)**0.5 * I2.dot(I2)**0.5)
        # theta_B1_I2 = np.arccos(B1.dot(I2)/
        #                         B1.dot(B1)**0.5 * I2.dot(I2)**0.5)
        # theta_A1_B1 = np.arccos(A1.dot(B1)/
        #                         A1.dot(A1)**0.5 * B1.dot(B1)**0.5)

        # I2_between_A1_B1 = abs(theta_A1_B1 - theta_A1_I2 - theta_B1_I2) < aatr  # noqa

        # # A2 B2 I2
        # theta_A2_I2 = np.arccos(A2.dot(I2)/
        #                         A2.dot(A2)**0.5 * I2.dot(I2)**0.5)
        # theta_B2_I2 = np.arccos(B2.dot(I2)/
        #                         B2.dot(B2)**0.5 * I2.dot(I2)**0.5)
        # theta_A2_B2 = np.arccos(A2.dot(B2)/
        #                         A2.dot(A2)**0.5 * B2.dot(B2)**0.5)

        # I2_between_A2_B2 = abs(theta_A2_B2 - theta_A2_I2 - theta_B2_I2) < aatr  # noqa

        # valid_int_for_I2 = I2_between_A1_B1 and I2_between_A2_B2

        # # hoping both of them will not be valid at the same time
        # # also lon_int and lat_int are initialised with NaNs
        # # and valid_int_flag with True
        # # refer: https://stackoverflow.com/a/1185413
        # # from xyz to lon lat, for a sphere
        # if valid_int_for_I1:
        #     lat_int[i] = np.arcsin(I1[2]) * 180/np.pi
        #     lon_int[i] = np.arctan2(I1[1], I1[0]) * 180/np.pi
        # elif valid_int_for_I2:
        #     lat_int[i] = np.arcsin(I2[2]) * 180/np.pi
        #     lon_int[i] = np.arctan2(I2[1], I2[0]) * 180/np.pi
        # else:
        #     valid_int_flag[i] = False

        # fortunately np.arctan2 returns longitude between -180 to 180

    return lon_shift(lon_int), lat_int, valid_int_flag


@jit(nopython=True, cache=True, parallel=True)
def simple_line_intersection_lonlat(x1a, y1a, x1b, y1b, x2a, y2a, x2b, y2b):
    # # Handling longitude value shifts at -180 and +180
    # checking longitudes shifts from 180 to -180 or vice versa
    lon_shift_flag_1 = np.abs(x1a - x1b) > lon_shift_condition_value
    lon_shift_flag_2 = np.abs(x2a - x2b) > lon_shift_condition_value

    # shifted the points that are crossing lon 180 to 0-360 range, i.e. around
    # +180. And then continue with calculations. Only for segments which do the
    # longitude crossing, intersection point longitudes will be above +180.
    # So do a lon shift at the end
    x1a = x1a + 360*(x1a < 0)*lon_shift_flag_1
    x1b = x1b + 360*(x1b < 0)*lon_shift_flag_1
    x2a = x2a + 360*(x2a < 0)*lon_shift_flag_2
    x2b = x2b + 360*(x2b < 0)*lon_shift_flag_2

    # the intersection calculations
    m1 = (y1b - y1a)/(x1b - x1a)
    m2 = (y2b - y2a)/(x2b - x2a)
    x_int = ((y2a - y1a) - (m2 * x2a - m1 * x1a))/(m1 - m2)
    y_int = m1 * x_int + y1a - m1 * x1a

    f = np.isinf(m1)  # first segment is vertical
    if f.sum() > 0:
        x_int[f] = x1a[f]  # which is same as x1b
        y_int[f] = y2a[f] + (y2b[f]-y2a[f]) * (x_int[f]-x2a[f])/(x2b[f]-x2a[f])

    f = np.isinf(m2)  # second segment is vertical
    if f.sum() > 0:
        x_int[f] = x2a[f]  # which is same as x2b
        y_int[f] = y1a[f] + (y1b[f]-y1a[f]) * (x_int[f]-x1a[f])/(x1b[f]-x1a[f])

    # what if both are parallel and there is not distance between the segments
    # This may cover the edge cases of both segments horizontal or both
    # segments vertical.
    p_f = m1 == m2  # parallel flag
    if p_f.sum() > 0:
        # no distance flag: if True then the two parallel segments do not have
        # any distance between each other
        nd_f = y1a + m1*(x2a - x1a) == y2a  # no distance flag
        f = np.logical_and(p_f, nd_f)
        if f.sum() > 0:
            # There is a common overlap between the parallel segments. So the
            # intersection point will be the mid-point of the overlapping parts
            # This is done by sorting all the four x values and y values and
            # getting the mean of the 2nd and 3rd values after sorting.
            x_int[f] = mean_ax1(sort_ax1_4(x1a[f], x1b[f], x2a[f], x2b[f])[:, 1:3])  # noqa
            y_int[f] = mean_ax1(sort_ax1_4(y1a[f], y1b[f], y2a[f], y2b[f])[:, 1:3])  # noqa

    # The common parallel condition should work for both as vertical and both
    # as horizontal
    # f = np.logical_and(np.isinf(m1), np.isinf(m2))  # both are vertical
    # if f.sum() > 0:
    #     x_int[f] = x1a[f]  # same as x1b, x2a, x2b
    #     y_int[f] = mean_ax1(sort_ax1_4(y1a[f], y1b[f], y2a[f], y2b[f])[:, 1:3])
    # f = np.logical_and((m1 == 0), (m2 == 0))  # both are horizontal
    # if f.sum() > 0:
    #     y_int[f] = y1a[f]  # same as y1b, y2a, y2b
    #     x_int[f] = mean_ax1(sort_ax1_4(x1a[f], x1b[f], x2a[f], x2b[f])[:, 1:3])

    # filtered out in get_recursive_box_indices()
    # # if any of these segments are just points, i.e. segments of length 0
    # # Then its not a crossover. This is to be compactable with GMT
    # f1 = np.logical_and((x1a == x1b), (y1a == y1b))
    # f2 = np.logical_and((x2a == x2b), (y2a == y2b))
    # f = np.logical_or(f1, f2)
    # if f.sum() > 0:
    #     x_int[f] = np.nan  # nan values will be filtered out at valid_int_flag
    #     y_int[f] = np.nan  # during the >=and <= conditions

    # the above calculation fails when slope is infinity
    # find where only one of the segment is vertical (not both of them)
    only_one_is_vertical = np.isinf(m1) ^ np.isinf(m2)
    if only_one_is_vertical.sum() > 0:
        for i in np.where(only_one_is_vertical)[0]:
            if np.abs(m1[i]) == np.inf:
                x_int[i] = x1a[i]  # which is same as x1b
                y_int[i] = y2a[i] + (y2b[i]-y2a[i]) * (x_int[i]-x2a[i])/(x2b[i]-x2a[i])  # noqa
            elif np.abs(m2[i]) == np.inf:
                x_int[i] = x2a[i]  # which is same as x2b
                y_int[i] = y1a[i] + (y1b[i]-y1a[i]) * (x_int[i]-x1a[i])/(x1b[i]-x1a[i])  # noqa

    # the above calculation fails when slope is zero
    # find where only one of the segment is horizontal (not both of them)
    only_one_is_horizontal = (m1 == 0) ^ (m2 == 0)
    if only_one_is_horizontal.sum() > 0:
        for i in np.where(only_one_is_horizontal)[0]:
            if np.abs(m1[i]) == 0:
                y_int[i] = y1a[i]  # which is same as y1b
            elif np.abs(m2[i]) == 0:
                y_int[i] = y2a[i]  # which is same as y2b

    # Exach coinsidence of segment ends. When this happens the intersection
    # formulae may not give the exact number that is equal to the segment ends
    # point 1a and point 2a exactly coinside
    a1_a2_f = np.logical_and((x1a == x2a), (y1a == y2a))
    x_int[a1_a2_f] = x1a[a1_a2_f]
    y_int[a1_a2_f] = y1a[a1_a2_f]
    # similarly for other combinations
    # pint 1a and point 2b
    a1_b2_f = np.logical_and((x1a == x2b), (y1a == y2b))
    x_int[a1_b2_f] = x1a[a1_b2_f]
    y_int[a1_b2_f] = y1a[a1_b2_f]
    # pint 1b and point 2a
    b1_a2_f = np.logical_and((x1b == x2a), (y1b == y2a))
    x_int[b1_a2_f] = x1b[b1_a2_f]
    y_int[b1_a2_f] = y1b[b1_a2_f]
    # pint 1b and point 2a
    b1_b2_f = np.logical_and((x1b == x2b), (y1b == y2b))
    x_int[b1_b2_f] = x1b[b1_b2_f]
    y_int[b1_b2_f] = y1b[b1_b2_f]

    # Edge_case
    # Pseudo segments check.
    # we are assigning NaNs here as intersection points to filter them out.
    # Assigning NaNs should be done at last so that they are not reasigned into
    # non NaN values because of other edge cases.
    # Psuedo segments were flagged out at get_recursive_box_indices but doing
    # the pseudo segment check again here for robust intersetcion function
    ps_f1 = np.logical_and((x1a == x1b), (y1a == y1b))  # pseudo segments
    ps_f2 = np.logical_and((x2a == x2b), (y2a == y2b))  # pseudo segments
    ps_f = np.logical_or(ps_f1, ps_f2)
    x_int[ps_f] = np.nan
    y_int[ps_f] = np.nan

    # Check if intersection points are on both lines and not in extrapolations
    # here the '=' helps when segments are completely horizontal or vertical
    # also checking for both y and x because because bugs where observed when a
    # segment is completely vertical and the = sign for x alone makes the
    # intersection valid even if intersection is exterior of segment. So both
    # x and y checks are required
    valid_int_flag = (
            (x_int <= max_ax1(x1a, x1b)) &
            (x_int >= min_ax1(x1a, x1b)) &
            (x_int <= max_ax1(x2a, x2b)) &
            (x_int >= min_ax1(x2a, x2b)) &
            (y_int <= max_ax1(y1a, y1b)) &
            (y_int >= min_ax1(y1a, y1b)) &
            (y_int <= max_ax1(y2a, y2b)) &
            (y_int >= min_ax1(y2a, y2b))
            )

    x_int[~valid_int_flag] = np.nan
    y_int[~valid_int_flag] = np.nan

    return lon_shift(x_int), y_int, valid_int_flag


# TODO: use this function to write more types of interpolations
# maybe polynomial or cubic spline
# Don't enable parallel unnecessarily
# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True, cache=True)
def interpolate(x_int, y_int, ls_idx, x, y, z):
    """
    float, float, int, float_array, float_array, float_array
    """
    # number of points on either side of the intersection point to be used
    # keeping it as 1 for simple linear interpolation
    n = 1

    s1 = ls_idx - n + 1
    s2 = s1 + 2*n
    s1 = 0 if s1 < 0 else s1

    d = distance_on_track(x[s1:s2], y[s1:s2])
    s = z[s1:s2]

    # value at this distance need to be found/interpolated
    d_int = distance_on_track(
            np.append(x[s1:s1+n], [x_int]),
            np.append(y[s1:s1+n], [y_int])
            )[-1]

    ######################
    # simple piece wise linear interpolation, since n = 1
    # change the following steps and n with what ever type of interpolation
    # you want. use d and s as xy data of fitting, and find the fit value
    # as d_int and update z_int
    m = (s[1] - s[0])/(d[1] - d[0])
    z_int = m*d_int + s[0]

    return z_int


# TODO
# Removing cache here because of get_recursive_box_indices
# Removing parallel because there is nothing to parallelise
@jit(nopython=True)
def cross_detect_with_internal(x1, y1, x2, y2,
                               bi_1=None, bi_2=None,
                               ps_f_1=None, ps_f_2=None,
                               min_x1=-np.inf, max_x1=np.inf,
                               min_y1=-np.inf, max_y1=np.inf,
                               min_x2=-np.inf, max_x2=np.inf,
                               min_y2=-np.inf, max_y2=np.inf,
                               rn=None,
                               recurse_level_limit=recurse_level_limit,
                               intersection_type="line",
                               internal_flag=True):

    ls1_idx, ls2_idx = get_recursive_box_indices(
            x1, y1, x2, y2,
            bi_1=bi_1, bi_2=bi_2,
            ps_f_1=ps_f_1, ps_f_2=ps_f_2,
            min_x1=min_x1, max_x1=max_x1,
            min_y1=min_y1, max_y1=max_y1,
            min_x2=min_x2, max_x2=max_x2,
            min_y2=min_y2, max_y2=max_y2,
            internal_flag=internal_flag,
            recurse_level_limit=recurse_level_limit,
            rn=rn)

    x1a = x1[:-1][ls1_idx]
    y1a = y1[:-1][ls1_idx]

    x2a = x2[:-1][ls2_idx]
    y2a = y2[:-1][ls2_idx]

    x1b = x1[1:][ls1_idx]
    y1b = y1[1:][ls1_idx]

    x2b = x2[1:][ls2_idx]
    y2b = y2[1:][ls2_idx]

    # this function needs lon lat in degrees, which is already the case
    if intersection_type == "line":
        (x_int, y_int,
         valid_int_flag) = simple_line_intersection_lonlat(x1a, y1a,
                                                           x1b, y1b,
                                                           x2a, y2a,
                                                           x2b, y2b)
    elif intersection_type == "arc":
        (x_int, y_int,
         valid_int_flag) = intersection_of_arcs_on_sphere(x1a, y1a,
                                                          x1b, y1b,
                                                          x2a, y2a,
                                                          x2b, y2b)

    x_int = x_int[valid_int_flag]  # filter invalid intersections
    y_int = y_int[valid_int_flag]
    ls1_idx = ls1_idx[valid_int_flag]
    ls2_idx = ls2_idx[valid_int_flag]

    # return n_int, x_int, y_int, ls1_idx, z1_int, d1, ls2_idx, z2_int, d2
    return x_int, y_int, ls1_idx, ls2_idx


# This is just combining cross_detect() and distance_on_track()
# @jit(nopython=True)
# @jit(nopython=True, parallel=True, cache=True)
# Removing cache here because cross_detect did not have it (because of
# get_recursive_box_indices)
@jit(nopython=True, parallel=True)
def cross_distances(lon1, lat1, z1, lon2, lat2, z2,
                    bi_1=None, bi_2=None,
                    d_1=None, d_2=None,
                    ps_f_1=None, ps_f_2=None,
                    min_x1=-np.inf, max_x1=np.inf,
                    min_y1=-np.inf, max_y1=np.inf,
                    min_x2=-np.inf, max_x2=np.inf,
                    min_y2=-np.inf, max_y2=np.inf,
                    rn=None,
                    recurse_level_limit=recurse_level_limit,
                    intersection_type="line",
                    internal_flag=True,
                    interpolation_type="simple"):

    x1 = lon1
    y1 = lat1
    x2 = lon2
    y2 = lat2
    (lon_int, lat_int, ls1_idx, ls2_idx
     ) = cross_detect_with_internal(
                    x1, y1, x2, y2,
                    bi_1=bi_1, bi_2=bi_2,
                    ps_f_1=ps_f_1, ps_f_2=ps_f_2,
                    min_x1=min_x1, max_x1=max_x1,
                    min_y1=min_y1, max_y1=max_y1,
                    min_x2=min_x2, max_x2=max_x2,
                    min_y2=min_y2, max_y2=max_y2,
                    rn=rn,
                    recurse_level_limit=recurse_level_limit,
                    internal_flag=internal_flag,
                    intersection_type=intersection_type)

    if d_1 is None:
        d_1 = distance_on_track(lon1, lat1)
    if d_2 is None:
        d_2 = distance_on_track(lon2, lat2)

    d_int1 = d_1[ls1_idx] + haversine(lon1[ls1_idx], lat1[ls1_idx],
                                      lon_int, lat_int)
    d_int2 = d_2[ls2_idx] + haversine(lon2[ls2_idx], lat2[ls2_idx],
                                      lon_int, lat_int)

    n_int = lon_int.size
    z1_int = np.zeros(n_int)
    z2_int = np.zeros(n_int)
    if interpolation_type == "simple":
        for i in prange(lon_int.size):
            z1_int[i] = interpolate(lon_int[i], lat_int[i], ls1_idx[i],
                                    lon1, lat1, z1)
            z2_int[i] = interpolate(lon_int[i], lat_int[i], ls2_idx[i],
                                    lon2, lat2, z2)

    return (n_int, lon_int, lat_int,
            ls1_idx, z1_int, d_int1,
            ls2_idx, z2_int, d_int2)


def fit_func_factory(d_self, fit_order, fundamental_distance, fit_margin):
    """
    returns (factory_fit_func, initial_parameters)
    first return value is a function, second is a numpy 1D array
    initial parameters are set to zero, and can primarily be used to give the
    number of unknown parameters (arguments) to scipy.optimize.curve_fit
    """
    n_xovers = d_self.size
    track_length = d_self.max() - d_self.min() if n_xovers > 1 else 0

    # only if there are enough crossovers to fit sinusoids
    if (n_xovers >= (4 + fit_margin)) and (fit_order >= 0):
        # What if track lenght is significantly shorter than fundamental lenght
        if track_length < fundamental_distance:
            individual_fundamental_distance = (
                 fundamental_distance / (fundamental_distance // track_length))

            required_distance_scale = (fundamental_distance/fit_order
                                       if fit_order > 0
                                       else fundamental_distance)

            individual_fit_order = (
              int(individual_fundamental_distance/required_distance_scale)
              if fit_order > 0 else fit_order)

        else:
            individual_fit_order = fit_order
            individual_fundamental_distance = fundamental_distance

        # What if the track lenght is long but no enough crossovers?
        # What if there are only 6 crossovers but individual_fit_order turned
        # out higher which requires atleast 14+fit_margin crossovers
        if (n_xovers - fit_margin) < (individual_fit_order*2 + 2):
            data_diff = (individual_fit_order*2 + 2) - (n_xovers - fit_margin)
            individual_fit_order -= int(np.ceil(data_diff/2))

        initial_parameters = np.zeros(individual_fit_order*2 + 2)

    elif (n_xovers > (1 + fit_margin) and
          n_xovers < (4 + fit_margin)) and (fit_order >= 0):
        initial_parameters = np.zeros(2)
        individual_fundamental_distance = fundamental_distance

    elif n_xovers <= (1 + fit_margin) or fit_order == -1:
        initial_parameters = np.zeros(1)
        individual_fundamental_distance = fundamental_distance

    else:
        # to capture other senearios (mainly n_xovers == 0) and have a function
        # and initial_parameters varialbes to return
        initial_parameters = np.zeros(0)
        individual_fundamental_distance = fundamental_distance

    # @jit(nopython=True, cache=True)
    def factory_fit_func(x, *args):
        """
        """
        nonlocal individual_fundamental_distance, d_self
        fd = individual_fundamental_distance
        arg_len = len(args)
        if arg_len == 0:  # if no aurguments are given
            out = x*0.0 + np.nan
        elif arg_len == 1:
            out = args[0] + 0*x  # bias
        elif arg_len == 2:
            out = args[0] + args[1] * x  # bias and slope
        elif arg_len > 2:  # sinusoids
            bias = args[0]
            slope = args[1]
            a = np.array(args[2::2])  # coeff of sin
            b = np.array(args[3::2])  # coeff of cos
            max_order = a.size

            out = bias + slope*x
            for order in np.arange(max_order)+1:
                # print(order)
                out += (a[order-1] * np.sin(order*(x/fd) * (2*np.pi)) +
                        b[order-1] * np.cos(order*(x/fd) * (2*np.pi)))

        return out

    return factory_fit_func, initial_parameters


def grid(lon, lat, ssh,
         lon_min=-180, lon_max=180,
         lat_min=-90, lat_max=90,
         lon_res=1, lat_res=1):

    lon_points = np.linspace(lon_min, lon_max,
                             int((lon_max-lon_min)/lon_res)+1)
    lat_points = np.linspace(lat_min, lat_max,
                             int((lat_max-lat_min)/lat_res)+1)

    lon_grid, lat_grid = np.meshgrid(lon_points, np.flip(lat_points))

    limit_flag = ((lon > lon_grid.min()) *
                  (lon < lon_grid.max()) *
                  (lat > lat_grid.min()) *
                  (lat < lat_grid.max()))
    lon = lon[limit_flag]
    lat = lat[limit_flag]
    ssh = ssh[limit_flag]

    # getting the resolutions of lat lon from points
    lon_res = abs(lon_points[1] - lon_points[0])
    lat_res = abs(lat_points[0] - lat_points[1])

    # removing any NaN values from the input data
    nan_flag = np.isnan(lon) | np.isnan(lat) | np.isnan(ssh)
    lon = lon[~nan_flag]
    lat = lat[~nan_flag]
    ssh = ssh[~nan_flag]

    # creating variables to store output with same dimensions as meshgrid
    # initialising as NaN
    ssh_grid = lon_grid + np.NaN
    std_grid = lon_grid + np.NaN
    no_data_grid = lon_grid + np.NaN

    lat_n, lon_n = lon_grid.shape
    for i in prange(lon_n):
        print(lon_grid[0, i])
        for j in prange(lat_n):
            lon_max = lon_grid[j, i] + lon_res/2
            lon_min = lon_grid[j, i] - lon_res/2
            lat_max = lat_grid[j, i] + lat_res/2
            lat_min = lat_grid[j, i] - lat_res/2

            flag = ((lon_min < lon) * (lon < lon_max) *
                    (lat_min < lat) * (lat < lat_max))

            no_data_grid[j, i] = flag.sum()
            ssh_grid[j, i] = ssh[flag].mean()
            std_grid[j, i] = np.std(ssh[flag])

    return lon_grid, lat_grid, ssh_grid, no_data_grid, std_grid


@jit(nopython=True, parallel=True, cache=True)
def get_hist_bins(x):
    """
    Refer: statisticshowto.com/choose-bin-sizes-statistics
    Refer: en.wikipedia.org/wiki/Histogram
    Survey has to be done regarding the right method applicable to us from the
    different methods mentioned in the article.
    For now using Sturges' formula.
    """
    gd_f = ~np.logical_or(np.isnan(x), np.isinf(x))  # good data flag
    n = gd_f.sum()
    bins = int(np.log(n)/np.log(2)) + 1

    current_range = np.nanmax(x) - np.nanmin(x)
    six_sigma_range = 6 * np.nanstd(x)

    bins = int(bins * current_range/six_sigma_range)
    return bins
