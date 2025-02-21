#!/usr/bin/env python
from multiprocessing import Pool
import pandas as pd
from numba import set_num_threads, jit
import sys
import os
sys.path.insert(0, os.getcwd())
from scipy.optimize import curve_fit  # noqa
from cross_over_module import (distance_on_track,  # noqa
                               fit_func_factory,
                               haversine,
                               lon_shift,
                               )
from config_cross import setup, calc  # noqa
# for disabling multi"threading" in numpy, in polyfit, since we are doing
# multiprocessing anyway. Should be set before importing numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np  # noqa


calc_id = sys.argv[1]
calc_obj = calc(calc_id)


detection_input_filename = calc_obj.detection_input_filename
post_correction_detection_filename = (
                              calc_obj.post_correction_detection_filename)
corrected_folder = calc_obj.corrected_folder
correcton_save_filename = calc_obj.correcton_save_filename

filename_match_pattern_self = calc_obj.filename_match_pattern_self
filename_match_pattern_other = calc_obj.filename_match_pattern_other
correction_weight = calc_obj.correction_weight

fundamental_distance = calc_obj.fundamental_distance
fit_order = calc_obj.fit_order

fit_margin = (calc_obj.fit_margin
              if hasattr(calc_obj, "fit_margin")
              else 0)


# distnace_limit_to_crossover = calc_obj.distnace_limit_to_crossover
# min_crossovers_in_distance_limit_eachside = (
#         calc_obj.min_crossovers_in_distance_limit_eachside)

if hasattr(calc_obj, "xdiff_max"):
    xdiff_max = calc_obj.xdiff_max
else:
    xdiff_max = np.inf

if hasattr(calc_obj, "xdiff_min"):
    xdiff_min = calc_obj.xdiff_min
else:
    xdiff_min = -np.inf

if hasattr(calc_obj, "interpolation_threshold"):
    interpolation_threshold = calc_obj.interpolation_threshold
else:
    interpolation_threshold = np.inf

if hasattr(calc_obj, "data_columnn_numbers"):
    data_columnn_numbers = calc_obj.data_columnn_numbers
elif hasattr(setup(), "data_columnn_numbers"):
    data_columnn_numbers = setup().data_columnn_numbers
else:
    data_columnn_numbers = [0, 1, 2]
print(f"data_columnn_numbers: {data_columnn_numbers}")

if hasattr(calc_obj, "input_data_delimiter"):
    input_data_delimiter = calc_obj.input_data_delimiter
elif hasattr(setup(), "input_data_delimiter"):
    input_data_delimiter = setup().input_data_delimiter
else:
    input_data_delimiter = r'\s+'
print(f'input_data_delimiter: {input_data_delimiter}')

curve_fit_maxfev = (calc_obj.curve_fit_maxfev
                    if hasattr(calc_obj, "curve_fit_maxfev")
                    else None)  # TODO find the correct default value

if hasattr(calc_obj, "number_of_processes"):
    number_of_processes = calc_obj.number_of_processes
elif hasattr(setup(), "number_of_processes"):
    number_of_processes = setup().number_of_processes
else:
    number_of_processes = 1
print(f"number_of_processes: {number_of_processes}")

if hasattr(calc_obj, "number_of_numba_threads"):
    number_of_numba_threads = calc_obj.number_of_numba_threads
elif hasattr(setup(), "number_of_numba_threads"):
    number_of_numba_threads = setup().number_of_numba_threads
else:
    number_of_numba_threads = 1

set_num_threads(number_of_numba_threads)

# If order is given as input argument then take that instead from config
fit_order = int(sys.argv[2]) if len(sys.argv) > 2 else fit_order


def file_read(name):
    print(f"reading {name}")
    data_i = np.asarray(
            pd.read_csv(name, header=None, sep=input_data_delimiter,
                        dtype=float))[:, data_columnn_numbers]
    data_i = np.c_[lon_shift(data_i[:, 0]), data_i[:, 1], data_i[:, 2]]
    return data_i


detection_df = pd.read_csv(detection_input_filename, sep="\t")

detection_df = detection_df.astype({"Track1":       "str",
                                    "Track2":       "str",
                                    "x_int":      "float",
                                    "y_int":      "float",
                                    "ls_index_1":   "int",
                                    "z1":         "float",
                                    "d1":         "float",
                                    "ls_index_2":   "int",
                                    "z2":         "float",
                                    "d2":         "float"})

filtered_detection_df = detection_df.loc[(
        (detection_df["Track1"].str.contains(filename_match_pattern_self) &
         detection_df["Track2"].str.contains(filename_match_pattern_other)) |
        (detection_df["Track1"].str.contains(filename_match_pattern_other) &
         detection_df["Track2"].str.contains(filename_match_pattern_self)))]

nan_flag = (np.isnan(filtered_detection_df["z1"]) |
            np.isnan(filtered_detection_df["z2"]))

filtered_detection_df = filtered_detection_df[~nan_flag]

tmp_flag1 = filtered_detection_df["Track1"].str.contains(
            filename_match_pattern_self)
tmp_series1 = filtered_detection_df["Track1"].loc[tmp_flag1]
tmp_flag2 = filtered_detection_df["Track2"].str.contains(
            filename_match_pattern_self)
tmp_series2 = filtered_detection_df["Track2"].loc[tmp_flag2]

names_a = np.unique(np.concatenate((tmp_series1, tmp_series2)))


del tmp_flag1, tmp_flag2, tmp_series1, tmp_series2


try:
    os.mkdir(corrected_folder)
except FileExistsError:
    print(f"{corrected_folder} already exists")


def main_func(fullfile_name):
    global fit_order, filtered_detection_df, corrected_folder
    self_detection_df = filtered_detection_df.loc[
            filtered_detection_df["Track1"].str.contains(fullfile_name) ^
            filtered_detection_df["Track2"].str.contains(fullfile_name)]

    # if our name is present as the second track instead of first track
    flip_flag = self_detection_df["Track2"].str.contains(fullfile_name)

    z_self = np.asarray(
            self_detection_df["z1"]*(1-flip_flag) +
            self_detection_df["z2"]*flip_flag)
    d_self = np.asarray(
            self_detection_df["d1"]*(1-flip_flag) +
            self_detection_df["d2"]*flip_flag)
    z_other = np.asarray(
            self_detection_df["z2"]*(1-flip_flag) +
            self_detection_df["z1"]*flip_flag)

    weight = correction_weight
    xdiff = (z_self-z_other)*weight

    xdiff_out_of_range_flag = (xdiff_min > xdiff) + (xdiff > xdiff_max)

    xdiff = xdiff[~xdiff_out_of_range_flag]
    d_self = d_self[~xdiff_out_of_range_flag]

    fit_func, initial_parameters = fit_func_factory(d_self, fit_order,
                                                    fundamental_distance,
                                                    fit_margin)

    try:
        popt, pcov = curve_fit(fit_func,
                               d_self, xdiff,
                               p0=initial_parameters,
                               maxfev=curve_fit_maxfev)
        fit_successful = True
        print(f"fitting: {fullfile_name} " +
              f"with {initial_parameters.size} parameters " +
              f"on {d_self.size} Xs")

    except ValueError:  # when no Xovers are left due to out of range xdiffs
        fit_successful = False
        print(f"No good crossovers: {fullfile_name} " +
              f"with {initial_parameters.size} parameters " +
              f"on {d_self.size} Xs")

    if fit_successful:

        optimisation_warn_flag = np.isnan(pcov).sum() > 0
        if optimisation_warn_flag:
            print(f"OptimizeWarning: {fullfile_name} " +
                  f"with {initial_parameters.size} parameters " +
                  f"on {d_self.size} Xs")

        data = file_read(fullfile_name)

        lon = data[:, 0]
        lat = data[:, 1]
        z = data[:, 2]

        distances = distance_on_track(lon, lat)

        corrections = fit_func(distances, *popt)
        original_corrections = np.copy(corrections)

        # # # Corrections happening as extrapolation of fit are not correct.
        # # # So check the parts which are not intrapolation of fit and
        # # # give NaNs
        # # interpolation_flag = ((distances < np.nanmax(d_self)) *
        # #                       (distances > np.nanmin(d_self)))
        # # corrections[~interpolation_flag] = np.nan

        # # When there are not enough crossovers in a region, fitting will be
        # # bad and should be excluded. Get a distance limit and check if
        # # there are minimum required number of crossovers on both sides of
        # # a data point within the distance limit

        # # See what's happening below. the 2d array has same rows as
        # # distances and same number of columns as d_self. Each element is
        # # for a combination
        # d_diff_2d = np.atleast_2d(d_self) - np.atleast_2d(distances).T

        # good_fit_flag_upper_side = (
        #         (d_diff_2d > -distnace_limit_to_crossover) *
        #         (d_diff_2d < 0)
        #         ).sum(axis=1) >= min_crossovers_in_distance_limit_eachside

        # good_fit_flag_lower_side = (
        #         (d_diff_2d < distnace_limit_to_crossover) *
        #         (d_diff_2d > 0)
        #         ).sum(axis=1) >= min_crossovers_in_distance_limit_eachside

        # good_fit_flag = good_fit_flag_upper_side * good_fit_flag_lower_side

        # corrections[~good_fit_flag] = np.nan

        # # Interpolation!
        d_self_diff = np.diff(np.sort(d_self))
        fwd_d_self_diff = np.append(d_self_diff, [np.nan])
        bwd_d_self_diff = np.append([np.nan], d_self_diff)

        fwd_large_d_self_flag = fwd_d_self_diff > interpolation_threshold
        bwd_large_d_self_flag = bwd_d_self_diff > interpolation_threshold
        # Note that nan > number is false

        fwd_interp = np.interp(distances,
                               np.sort(d_self), fwd_large_d_self_flag)
        bwd_interp = np.interp(distances,
                               np.sort(d_self), bwd_large_d_self_flag)
        # flags will be interpolated between 0 and 1

        interp_flag = (fwd_interp != 0) * (bwd_interp != 0)

        corrections[interp_flag] = np.interp(
                distances[interp_flag],
                np.sort(d_self),
                fit_func(np.sort(d_self), *popt))

        # # Extrapolation!
        left_extra_flag = distances < d_self.min()
        right_extra_flag = distances > d_self.max()
        corrections[left_extra_flag] = fit_func(d_self.min(), *popt)
        corrections[right_extra_flag] = fit_func(d_self.max(), *popt)

        z_corrected = z - corrections

        piece_wise_interpol = np.interp(distances,
                                        d_self[d_self.argsort()],
                                        xdiff[d_self.argsort()])

        corrected_data = np.c_[lon, lat, z_corrected,
                               z, piece_wise_interpol,
                               original_corrections,
                               interp_flag, left_extra_flag + right_extra_flag]

        corrected_name = (corrected_folder +
                          fullfile_name.split("/")[-1])

        np.savetxt(corrected_name,
                   corrected_data,
                   fmt="%14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %i %i")

        tmp_fit_df = pd.DataFrame({"name": [fullfile_name]})

        tmp_fit_df["optimisation_warn"] = optimisation_warn_flag

        tmp_fit_df["min_corr"] = np.nanmin(corrections)
        tmp_fit_df["min_xdiff"] = np.nanmin(xdiff)

        tmp_fit_df["max_corr"] = np.nanmax(corrections)
        tmp_fit_df["max_xdiff"] = np.nanmax(xdiff)

        tmp_fit_df["max_abs_dev"] = np.nanmax(np.abs(
                corrections -
                piece_wise_interpol))

        if interp_flag.sum() > 0:
            tmp_fit_df["max_abs_interp_dev"] = np.nanmax(np.abs(
                    corrections[interp_flag] -
                    original_corrections[interp_flag]))
        else:
            tmp_fit_df["max_abs_interp_dev"] = 0

        for i in range(popt.size):
            tmp_fit_df[f"coeff_{i}"] = popt[i]

        # print(tmp_fit_df)
        return tmp_fit_df


P = Pool(processes=number_of_processes)
print("created a pool, doing calculations now")

result_list = P.map(main_func, names_a)

print("fitting operations done")

print("closing pool now")
P.close()
P.join()

# result_list is a list of data frames each with one row corresponding to one
# track. Also this list contains None values when polyfit is not done for a
# track corrections_df.append(result_list) doesn't work if the first value of
# result_list is None. To append these into a single dataframe a dummy
# dataframe is added in the beginning to give the right columns during
# appending.

# Also this doesn't work : result_list = list(filter(None, result_list))
# Because truth value of a dataframe is not defined

dummy_df = pd.DataFrame({"name": ["dummy"]})  # First column
dummy_df["optimisation_warn"] = False
dummy_df["min_corr"] = np.nan
dummy_df["min_xdiff"] = np.nan
dummy_df["max_corr"] = np.nan
dummy_df["max_xdiff"] = np.nan
dummy_df["max_abs_dev"] = np.nan
dummy_df["max_abs_interp_dev"] = np.nan
for i in range(fit_order*2 + 2):
    dummy_df[f"coeff_{i}"] = np.nan  # other columns

result_list.insert(0, dummy_df)

corrections_df = pd.DataFrame()
corrections_df = corrections_df.append(result_list, ignore_index=True)

# Dropping the first row (dummy)
corrections_df = corrections_df.iloc[1:, :]

# reset_index, or else index will start with 1 once file is saved (heresy!)
corrections_df = corrections_df.reset_index(drop=True)

corrections_df.to_csv(correcton_save_filename, sep="\t", na_rep="NaN")


print("Good bye")

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# using np.copy instead of np.array or else detection_df (pandas dataframe) and
# variables (numpy arrays) will use the same memory and "connected". Since
# these vaiables will be overwritten, it will also effect detection_df
# Yup, python being python
# check: https://stackoverflow.com/questions/36285916

# pls don't give a dtype here as str. right now its considered as "object"
# there is an issue during array broadcasting later
track1_ffn_a = np.copy(detection_df["Track1"])
track2_ffn_a = np.copy(detection_df["Track2"])

lon_int_a = np.copy(detection_df["x_int"])
lat_int_a = np.copy(detection_df["y_int"])
ls1_idx_a = np.copy(detection_df["ls_index_1"])
ls2_idx_a = np.copy(detection_df["ls_index_2"])
d1_a = np.copy(detection_df["d1"])
d2_a = np.copy(detection_df["d2"])
z1_a = np.copy(detection_df["z1"])
z2_a = np.copy(detection_df["z2"])


corrected_fn_a = np.array([full_name.split("/")[-1]
                           for full_name in corrections_df["name"]])

track1_fn_a = np.array([full_name.split("/")[-1]
                        for full_name in track1_ffn_a])

track2_fn_a = np.array([full_name.split("/")[-1]
                        for full_name in track2_ffn_a])


corrected_flag1 = np.isin(track1_fn_a, corrected_fn_a)
corrected_flag2 = np.isin(track2_fn_a, corrected_fn_a)

change_flag = corrected_flag1 | corrected_flag2

# for files that were corrected, use the file path of the new corrected file
track1_ffn_a[
        np.nonzero(corrected_flag1)
        ] = np.array([corrected_folder + "/" + fn
                      for fn in track1_fn_a[corrected_flag1]], dtype=str)

track2_ffn_a[
        np.nonzero(corrected_flag2)
        ] = np.array([corrected_folder + "/" + fn
                      for fn in track2_fn_a[corrected_flag2]], dtype=str)


# To get the right data type
# I don't know WTF is happening without this when given as input to numba
# function. So I am converting a numpy string array to list and then numpy
# array again!
track1_ffn_a = np.array(list(track1_ffn_a))
track2_ffn_a = np.array(list(track2_ffn_a))


# nr: non repeating tracks array or list. Because same data file should not be
# read multiple times
# np.unique sorts the array, so file names are again calculated from ffn, i.e.
# full file name. This cannot be done by concatenateing track1_fn_a and
# track2_fn_a since the sorting order is going to be different.
nr_tracks_ffn_a = np.unique(np.concatenate((track1_ffn_a[change_flag],
                                            track2_ffn_a[change_flag])))
nr_tracks_fn_a = np.array([full_name.split("/")[-1]
                          for full_name in nr_tracks_ffn_a])




# # Normal reading
# nr_data_list = [file_read(name) for name in nr_tracks_ffn_a]

P = Pool(processes=number_of_processes)
print("created a Pool of processes, for reading data")

nr_data_list = P.map(file_read, nr_tracks_ffn_a)

print("closing data reading pool")
P.close()
P.join()


@jit(nopython=True, cache=True)
def get_index(name, nr_track_names_a):
    for i in range(nr_track_names_a.size):
        if name == nr_track_names_a[i]:
            return i


# lon1=lon1, lat1=lat1, z1=z1, lon2=lon2, lat2=lat2, z2=z2):

@jit(nopython=True, cache=True)
def single_row(i,
               lon_int_a, lat_int_a, ls1_idx_a, ls2_idx_a,
               lon1, lat1, z1, lon2, lat2, z2):

    # lon1 = track1_data[:, 0]
    # lat1 = track1_data[:, 1]
    # z1 = track1_data[:, 2]

    # lon2 = track2_data[:, 0]
    # lat2 = track2_data[:, 1]
    # z2 = track2_data[:, 2]

    lon_int = lon_int_a[i]
    lat_int = lat_int_a[i]
    ls1_idx = ls1_idx_a[i]
    ls2_idx = ls2_idx_a[i]

    # print(lon_int, lat_int, ls1_idx, ls2_idx)

    # nan_flag1 = np.isnan(lon1) | np.isnan(lat1)
    # lon1 = lon1[~nan_flag1]
    # lat1 = lat1[~nan_flag1]
    # z1 = z1[~nan_flag1]

    # nan_flag2 = np.isnan(lon2) | np.isnan(lat2)
    # lon2 = lon2[~nan_flag2]
    # lat2 = lat2[~nan_flag2]
    # z2 = z2[~nan_flag2]

    lon1a = lon1[ls1_idx]
    lat1a = lat1[ls1_idx]
    lon2a = lon2[ls2_idx]
    lat2a = lat2[ls2_idx]

    lon1b = lon1[ls1_idx + 1]
    lat1b = lat1[ls1_idx + 1]
    lon2b = lon2[ls2_idx + 1]
    lat2b = lat2[ls2_idx + 1]

    z1a = z1[ls1_idx]
    z2a = z2[ls2_idx]

    z1b = z1[ls1_idx+1]
    z2b = z2[ls2_idx+1]

    # print(lon1a, lat1a, z1a, lon2a, lat2a, z2a)

    # For ratio of p:q for the distances da, db  between points A,int_point
    # and int_point,B, z_int = (da*q + db*p)/(p+q)
    # Here we are taking the great circle distance.
    # Radius of planet is given constant since it will cancel out in the
    # ratio
    d1a = haversine(lon1a, lat1a, lon_int, lat_int)
    d1b = haversine(lon1b, lat1b, lon_int, lat_int)
    d2a = haversine(lon2a, lat2a, lon_int, lat_int)
    d2b = haversine(lon2b, lat2b, lon_int, lat_int)
    new_z1_int = (z1a*d1b + z1b*d1a)/(d1a + d1b)
    new_z2_int = (z2a*d2b + z2b*d2a)/(d2a + d2b)

    return np.array([new_z1_int, new_z2_int])


def wrapper(i,
            track1_fn_a=track1_fn_a,
            track2_fn_a=track2_fn_a,
            lon_int_a=lon_int_a,
            lat_int_a=lat_int_a,
            ls1_idx_a=ls1_idx_a,
            ls2_idx_a=ls2_idx_a,
            nr_tracks_fn_a=nr_tracks_fn_a):
    """
    reading data is done in a wrapper since taking data from a list of 2d
    arrays (3 cols and rows for each data file) has issues.
    Like they have to be done in a object mode context manager, and even with
    that cacheing cannot be done.
    """
    track1_fn = track1_fn_a[i]
    track2_fn = track2_fn_a[i]
    # print(track1_fn, track2_fn)

    data_idx_1 = get_index(track1_fn, nr_tracks_fn_a)
    data_idx_2 = get_index(track2_fn, nr_tracks_fn_a)

    track1_data = nr_data_list[data_idx_1]
    track2_data = nr_data_list[data_idx_2]

    lon1 = track1_data[:, 0]
    lat1 = track1_data[:, 1]
    z1 = track1_data[:, 2]

    lon2 = track2_data[:, 0]
    lat2 = track2_data[:, 1]
    z2 = track2_data[:, 2]
    # print(data_idx_1)

    print(f'\rupdating row : {i}                     ', end='')

    return single_row(i,
                      lon_int_a, lat_int_a, ls1_idx_a, ls2_idx_a,
                      lon1, lat1, z1, lon2, lat2, z2)


P = Pool(processes=number_of_processes)
print("created a pool, doing calculations now")

change_idx, = np.where(change_flag)
print(f'{change_flag.sum()} rows should be updated from {change_flag.size} in total')  # noqa
result_list = P.map(wrapper, change_idx)

print('')
print("closing pool now")
P.close()
P.join()

columnised_result = np.c_[result_list]

new_z1_a = z1_a
new_z2_a = z2_a
new_z1_a[change_idx] = columnised_result[:, 0]
new_z2_a[change_idx] = columnised_result[:, 1]

# TODO: Don't create a new post_correction_df. copy the detection_df and 
# update the columns
post_correction_df = pd.DataFrame({"Track1":     track1_ffn_a,
                                   "Track2":     track2_ffn_a,
                                   "x_int":      lon_int_a,
                                   "y_int":      lat_int_a,
                                   "ls_index_1": ls1_idx_a,
                                   "z1":         new_z1_a,
                                   "d1":         d1_a,
                                   "ls_index_2": ls2_idx_a,
                                   "z2":         new_z2_a,
                                   "d2":         d2_a})

post_correction_df = post_correction_df.astype({"Track1":       "str",
                                                "Track2":       "str",
                                                "x_int":      "float",
                                                "y_int":      "float",
                                                "ls_index_1":   "int",
                                                "z1":         "float",
                                                "d1":         "float",
                                                "ls_index_2":   "int",
                                                "z2":         "float",
                                                "d2":         "float"})

post_correction_df.to_csv(post_correction_detection_filename,
                          sep="\t", na_rep="NaN")


###############################################################################
###############################################################################
###############################################################################
###############################################################################

fit_explosion_s = corrections_df["max_abs_dev"]

# _s prefix here denotes that the variable is a pandas series
change_flag_s = (
        post_correction_df["Track1"].str.contains(corrected_folder) |
        post_correction_df["Track2"].str.contains(corrected_folder))


# erm_match_string = re.sub("/", "", erm_data_folder)


required_crossover_flag_s = (
    (detection_df["Track1"].str.contains(filename_match_pattern_self) &
     detection_df["Track2"].str.contains(filename_match_pattern_other)) |
    (detection_df["Track1"].str.contains(filename_match_pattern_other) &
     detection_df["Track2"].str.contains(filename_match_pattern_self)))

flip_flag = (
      detection_df["Track1"].str.contains(filename_match_pattern_other) &
      detection_df["Track2"].str.contains(filename_match_pattern_self))

uncorrected_differences_s = (
        detection_df["z1"] - detection_df["z2"]
        ).loc[required_crossover_flag_s & change_flag_s]
uncorrected_differences_s[flip_flag[change_flag_s]] *= -1
uncorrected_differences = np.asarray(uncorrected_differences_s)

corrected_differences_s = (
        post_correction_df["z1"] - post_correction_df["z2"]
        ).loc[required_crossover_flag_s & change_flag_s]
corrected_differences_s[flip_flag[change_flag_s]] *= -1
corrected_differences = np.asarray(corrected_differences_s)

x_int = detection_df["x_int"].loc[
        required_crossover_flag_s & change_flag_s]
y_int = detection_df["y_int"].loc[
        required_crossover_flag_s & change_flag_s]

# Note that number of nans and their locations should be same for corrected
# and uncorrected differences
corrected_differences = corrected_differences[~np.isnan(corrected_differences)]
uncorrected_differences = uncorrected_differences[
        ~np.isnan(uncorrected_differences)]

internal_crossover_flag_s = (
        post_correction_df["Track1"].str.contains(
                                        filename_match_pattern_self) &
        post_correction_df["Track2"].str.contains(
                                        filename_match_pattern_self))

internal_change_flag_s = (
        post_correction_df["Track1"].str.contains(corrected_folder) &
        post_correction_df["Track2"].str.contains(corrected_folder))

internal_uc_diff_s = (
        detection_df["z2"] - detection_df["z1"]
        ).loc[internal_crossover_flag_s & change_flag_s].abs()
internal_c_diff_s = (
        post_correction_df["z2"] - post_correction_df["z1"]
        ).loc[internal_crossover_flag_s & change_flag_s].abs()

internal_x_int = post_correction_df["x_int"].loc[
                                     internal_crossover_flag_s & change_flag_s]
internal_y_int = post_correction_df["y_int"].loc[
                                     internal_crossover_flag_s & change_flag_s]

no_hist_bins = 100
hist_min = -0.5
hist_max = 0.5

uncorrected_hist, x = np.histogram(uncorrected_differences,
                                   range=(hist_min, hist_max),
                                   bins=no_hist_bins)
corrected_hist, x = np.histogram(corrected_differences,
                                 range=(hist_min, hist_max),
                                 bins=no_hist_bins)
x = x[:-1] + np.diff(x)/2


def gaussian(x, *a):
    a0, a1, a2, a3 = a
    return a0 + a1*np.exp(-((x-a3)/a2)**2)


def my_rmse(corrected_differences):
    return np.sqrt(np.sum(corrected_differences**2)/corrected_differences.size)


def my_std(corrected_differences):
    corrected_differences = (
            corrected_differences - corrected_differences.mean())
    return np.sqrt(np.sum(corrected_differences**2)/corrected_differences.size)


original_stdout = sys.stdout
with open("./output_statistic.txt", "a") as f:
    sys.stdout = f
    print(
        calc_id,
        fit_order,
        corrected_differences.size,
        np.nanmean(uncorrected_differences), np.nanmean(corrected_differences),
        my_std(uncorrected_differences), my_std(corrected_differences),
        my_rmse(uncorrected_differences), my_rmse(corrected_differences),
        np.nanmin(uncorrected_differences), np.nanmin(corrected_differences),
        np.nanmax(uncorrected_differences), np.nanmax(corrected_differences),
        np.nanmean(internal_uc_diff_s), np.nanmean(internal_c_diff_s),
        my_std(internal_uc_diff_s), my_std(internal_c_diff_s),
        my_rmse(internal_uc_diff_s), my_rmse(internal_c_diff_s),
        internal_uc_diff_s.min(), internal_c_diff_s.min(),
        internal_uc_diff_s.max(), internal_c_diff_s.max(),
        np.mean(fit_explosion_s), np.std(fit_explosion_s),
        np.max(fit_explosion_s)
        )
    sys.stdout = original_stdout

# initial_a = [0, 1, 2, 0]

# corrected_coeff, _ = curve_fit(gaussian, x, corrected_hist, initial_a)
# uncorrected_coeff, _ = curve_fit(gaussian, x, uncorrected_hist, initial_a)

# print("# calc_id fit_order size_of_differences \
# mean_uncorrected mean_corrected \
# std_uncorrected std_corrected \
# rmse_uncorrected rmse_corrected \
# min_uncorrected min_corrected \
# max_uncorrected max_corrected")


# # plt.figure(0)
# plt.scatter(x, uncorrected_hist, alpha=0.5)
# plt.plot(x, gaussian(x, *uncorrected_coeff),
#          label=("Uncorrected\n" +
#                 f"std={uncorrected_coeff[2]:.3f}\n" +
#                 f"mean={uncorrected_coeff[3]:.3f}"))
# plt.scatter(x, corrected_hist, alpha=0.5)
# plt.plot(x, gaussian(x, *corrected_coeff),
#          label=("corrected\n" +
#                 f"std={corrected_coeff[2]:.3f}\n" +
#                 f"mean={corrected_coeff[3]:.3f}"))

# # plt.xlabel("SSH difference (m)")
# # plt.ylabel("Number of occurrences")
# # plt.title(f"\
# # Region {calc_id}, Order {fit_order}, on \
# # {corrected_differences.size} crossovers\n \
# # RMSE {my_rmse(uncorrected_differences):.2f} before and \
# # {my_rmse(corrected_differences):.2f} m after correction")
# # plt.legend()

# # plt.savefig(f"../images/{calc_id}_statistic_{fit_order}.png",
# #             )
# #             # transparent=True)
# # plt.show(block=False)

# gs = gridspec.GridSpec(2, 2,
#                        height_ratios=[2, 1],
#                        width_ratios=[1, 1])

# fig = plt.figure(f"{calc_id} {fit_order} comparision")
# ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
# ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# # ax1.set_adjustable("datalim")
# ax2.get_shared_x_axes().join(ax2, ax1)
# ax2.get_shared_y_axes().join(ax2, ax1)

# ax1.coastlines("10m")
# ax2.coastlines("10m")

# ax1.scatter(
#         x_int, y_int, c=uncorrected_differences_s,
#         transform=ccrs.PlateCarree(),
#         vmin=-0.1, vmax=0.1, s=15, cmap='coolwarm')

# ax2_scatter = ax2.scatter(
#         x_int, y_int, c=corrected_differences_s,
#         transform=ccrs.PlateCarree(),
#         vmin=-0.1, vmax=0.1, s=15, cmap='coolwarm')

# fig.colorbar(ax2_scatter, ax=[ax1, ax2], orientation="horizontal",
#              extend="both")
