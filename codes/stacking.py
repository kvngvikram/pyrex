#!/usr/bin/env python
import numpy as np
from cross_over_module import (haversine, planet_radius_simple_ellipsoid,
                               lon_shift)
from numba import njit, set_num_threads
from multiprocessing import Pool
import pandas as pd
import sys
import glob

from config_cross import setup, calc

calc_id = sys.argv[1]
calc_obj = calc(calc_id)

number_of_tracks = calc_obj.number_of_tracks
cutoff = calc_obj.cutoff
stacking_file_read_glob_f_string_list = (
                         calc_obj.stacking_file_read_glob_f_string_list)
stacking_save_filepaths_f_string = (
                             calc_obj.stacking_save_filepaths_f_string)

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
print(f"number_of_numba_threads: {number_of_numba_threads}")

max_iter = (calc_obj.max_iter
            if hasattr(calc_obj, "max_iter")
            else 20)

data_skip_value = (calc_obj.data_skip_value
                   if hasattr(calc_obj, "data_skip_value")
                   else 1)

longest_track_name_match_pattern = (
        calc_obj.longest_track_name_match_pattern
        if hasattr(calc_obj, "longest_track_name_match_pattern")
        else '')
tracks_for_averaging_name_match_pattern = (
        calc_obj.tracks_for_averaging_name_match_pattern
        if hasattr(calc_obj, "tracks_for_averaging_name_match_pattern")
        else '')

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


def file_reading(i_track):

    file_list = []
    for read_glob_f_sting in stacking_file_read_glob_f_string_list:
        file_list += sorted(glob.glob(
                eval("f\"{}\"".format(read_glob_f_sting))
                ))

    data_list = []
    size_list = []
    i = 0  # used for numbering files, file numbering starts with 0
    cycle_indices_array_list = []
    for filename in file_list:
        try:
            tmp_np2d_array = np.atleast_2d(
                    pd.read_csv(filename, header=None,
                                sep=input_data_delimiter))[:, data_columnn_numbers]  # noqa
            data_list.append(tmp_np2d_array)
            size = (~np.isnan(tmp_np2d_array[:, 0])).sum()
            size_list.append(size)
            cycle_indices_array_list.append(np.ones(tmp_np2d_array.shape[0])*i)
            i += 1
        except pd.errors.EmptyDataError:
            print(f"empty file: {filename}")
            i += 1
            size_list.append(0)
            data_list.append(None)

    matched_for_longest_f = pd.Series(
            file_list).str.contains(longest_track_name_match_pattern)

    max_data_arg = np.arange(len(file_list))[
            matched_for_longest_f
        ][np.argmax(np.asarray(size_list)[matched_for_longest_f])]

    lon_stk_1 = data_list[max_data_arg][:, 0]
    lat_stk_1 = data_list[max_data_arg][:, 1]

    lon_stk_1 = lon_stk_1[::data_skip_value]
    lat_stk_1 = lat_stk_1[::data_skip_value]

    print(f"Mean track : {file_list[max_data_arg]} " +
          f"has {size_list[max_data_arg]} points")

    matched_for_avg_f = pd.Series(
            file_list).str.contains(tracks_for_averaging_name_match_pattern)

    data_list_for_avg = [data_list[i] for i in np.where(matched_for_avg_f)[0]]

    erm_data = np.concatenate(tuple(data_list_for_avg), axis=0)
    cycle_indices = np.concatenate(tuple(cycle_indices_array_list))
    lon_erm, lat_erm, ssh_erm = erm_data[:, 0], erm_data[:, 1], erm_data[:, 2]

    return (lon_shift(lon_erm), lat_erm, ssh_erm, cycle_indices,
            lon_shift(lon_stk_1), lat_stk_1)


set_num_threads(number_of_numba_threads)

# Just to supress a warning :|
cutoff = cutoff
max_iter = max_iter


@njit(cache=True)
def weights_func(distance_from_target):
    global cutoff

    # This function is basically giving weights for averaging points within a
    # distance of cutoff, basically the cutoff is the radius of circle.
    # This function should be customised as required but be careful with some
    # conventions to used.
    # The weights should be normalised to 1, but if all weights are 0 then give
    # nans for all values and no negative values
    # taking only one input argument and any other inputs taken from global
    # variables

    weights = np.zeros(distance_from_target.shape)
    weights[distance_from_target < cutoff] = 1
    nan_flag = np.isnan(weights)
    weights[nan_flag] = 0  # removing any nan values
    all_zero_flag = (weights == 0).sum() == weights.size
    # return nan array if all the weights are zero or else a normalised array
    return weights + np.nan if all_zero_flag else weights/weights.sum()


@njit(cache=True)
def get_not_outlier_flag(ssh):
    global max_iter

    ssh_mean = np.nanmean(ssh)
    ssh_std = np.nanstd(ssh)

    flag = ((ssh < ssh_mean + 3*ssh_std) * (ssh > ssh_mean - 3*ssh_std))

    new_ssh_mean = np.nanmean(ssh[flag])
    new_ssh_std = np.nanstd(ssh[flag])

    i = 1

    while new_ssh_std < ssh_std and i < max_iter:
        ssh_mean = new_ssh_mean
        ssh_std = new_ssh_std

        flag = (flag *
                ((ssh < ssh_mean + 3*ssh_std) * (ssh > ssh_mean - 3*ssh_std)))

        new_ssh_mean = np.nanmean(ssh[flag])
        new_ssh_std = np.nanstd(ssh[flag])

        i += 1

    return flag, i


# don't paralallise this. Don't know why but results are wrong, inf, or nan
# @njit(cache=True)
@njit
def per_track_numba(lon_stk_1, lat_stk_1,
                    lon_erm, lat_erm, ssh_erm, cycle_indices):
    n_records = lon_stk_1.size

    # initialising a nan array, and nan may be replaced with values later
    ssh = np.zeros(n_records) * np.nan

    # initialising arrays. 0s or nans may be replaced with values later
    number_of_data_averaged = np.zeros(n_records)
    number_of_outlier_removal_iterations = np.zeros(n_records)
    std_of_averaged_data = np.zeros(n_records) * np.nan
    number_of_cycles_stacked = np.zeros(n_records)

    # This margin is used to filter out points that are decently farther than
    # the cutoff distance. This is basically used to filter out data and reduce
    # the computational load (also make it faster). This will be a courser
    # cutoff instead of calculating haversine distance.
    # This is only for latitude as distance between between two longitudes can
    # vary with latitude and this doesn't need to be precise.
    # giving 0.0 to give numba some clarity about datatype in future use :P
    equitorial_radius = planet_radius_simple_ellipsoid(0.0)
    lat_angle_subtended_by_cutoff = (cutoff/equitorial_radius)*180/np.pi
    closeness_margin_lat = lat_angle_subtended_by_cutoff * 1.01

    for i_record in range(n_records):
        # local_planet_radius = planet_radius_simple_ellipsoid(
        #         lat_stk_1[i_record])

        lon_angle_subtended_by_cutoff = (
                (cutoff/equitorial_radius) /
                np.cos(lat_stk_1[i_record] * np.pi/180)) * 190/np.pi
        closeness_margin_lon = lon_angle_subtended_by_cutoff * 1.01

        close_enough_flag = (
                (lat_stk_1[i_record] + closeness_margin_lat > lat_erm) *
                (lat_stk_1[i_record] - closeness_margin_lat < lat_erm) *
                (lon_stk_1[i_record] + closeness_margin_lon > lon_erm) *
                (lon_stk_1[i_record] - closeness_margin_lon < lon_erm))

        distance_from_target = haversine(lon_erm[close_enough_flag],
                                         lat_erm[close_enough_flag],
                                         lon_stk_1[i_record],
                                         lat_stk_1[i_record],
                                         )

        weights = weights_func(distance_from_target)  # already normalised

        # Here the weights are already normalised but if there are nan values
        # in ssh_erm for which weights are non zero then only few of the
        # weights will be used and for the rest with nans in ssh_erm the
        # multiplication will be nan. Weights overall are normalized but if
        # only few of the weights are used then normalisation should be done
        # again

        flag = weights > 0

        inside_cutoff_ssh = ssh_erm[close_enough_flag][flag]
        inside_cutoff_weights = weights[flag]

        not_outlier_flag, iter_done = get_not_outlier_flag(inside_cutoff_ssh)

        selected_ssh = inside_cutoff_ssh[not_outlier_flag]
        selected_weights = inside_cutoff_weights[not_outlier_flag]

        tmp_multiple = selected_ssh * selected_weights

        # basically flag if ssh values have nans
        nan_flag = np.isnan(tmp_multiple)

        # re normaize again with the left over weights after all filterations
        tmp_multiple = tmp_multiple/selected_weights[~nan_flag].sum()

        number_of_data_averaged[i_record] = (~nan_flag).sum()

        if number_of_data_averaged[i_record] > 0:
            ssh[i_record] = np.nansum(tmp_multiple)

        average = ssh[i_record]

        # For standard deviation
        tmp_multiple = (selected_ssh - average)**2 * selected_weights

        # nan_flag can again be calculated from tmp_multiple here but will be
        # same as before
        tmp_multiple = tmp_multiple/selected_weights[~nan_flag].sum()

        if number_of_data_averaged[i_record] > 1:
            std_of_averaged_data[i_record] = np.sqrt(np.nansum(tmp_multiple))

        cycle_indices_left = (
                cycle_indices[close_enough_flag][flag]
                                                )[not_outlier_flag][~nan_flag]

        number_of_cycles_stacked[i_record] = np.unique(cycle_indices_left).size
        number_of_outlier_removal_iterations[i_record] = iter_done

        # print(i_record)
        # print(weights)
        # print(average, number_of_data_averaged[i_record],
        #       std_of_averaged_data[i_record])

        # print(i_record)
    # return value will be a 2D array
    return np.column_stack((ssh,
                            number_of_data_averaged,
                            std_of_averaged_data,
                            number_of_outlier_removal_iterations,
                            number_of_cycles_stacked))


def wrapper_per_track_numba(i_track):
    print(f"file reading : {i_track+1:03d}")
    (lon_erm, lat_erm, ssh_erm, cycle_indices,
     lon_stk_1, lat_stk_1) = file_reading(i_track)

    print(f"doing track : {i_track+1:03d}")

    out = per_track_numba(lon_stk_1, lat_stk_1,
                          lon_erm, lat_erm, ssh_erm, cycle_indices)

    save_filename = eval(
            "f\"{}\"".format(stacking_save_filepaths_f_string))
    print(f"saving file {save_filename}")
    np.savetxt(save_filename,
               np.c_[
                   lon_stk_1, lat_stk_1, out],
               fmt="%12.8f %12.8f %12.8f %8d %12.8f %8d %8d")

    print(f"done track : {i_track+1:03d}")

    return out


# i_track = 68
# result = wrapper_per_track_numba(i_track)

# i_track = 346
n_tracks = number_of_tracks

P = Pool(processes=number_of_processes)
# P = Pool(processes=2)
print("created a pool, doing calculations now")

# result_list = P.map(wrapper_per_track_numba, range(n_tracks))
P.map(wrapper_per_track_numba, range(n_tracks))
# P.map(wrapper_per_track_numba, range(2))

# result_list here is a list of 2D arrays for each track  with rows for records
# of each track and columns for ssh, no data, and std

print("calculations complete!")

P.close()
P.join()

# # These following lines are commented but not deleted for reference purposes
# # Here the result_list is a list of tuple (3 element) of 1d numpy arrays
# # the three elements in each tuple are ssh 1d array, no points 1d array and
# # std 1d array
# # Flattening result_list is pickedup from
# # https://stackoverflow.com/a/10636583
# flat_result_list = list(sum(result_list, ()))
# ssh = np.c_[flat_result_list[0::3]]
# number_of_data_averaged = np.c_[flat_result_list[1::3]]
# std_of_averaged_data = np.c_[flat_result_list[2::3]]

# for i_track in range(n_tracks):
#     filename = eval(
#             "f\"{}\"".format(js2_stacking_save_filepaths_f_string))
#     print(f"saving file {filename}")
#     np.savetxt(filename,
#                np.c_[
#                    lon_stk[i_track], lat_stk[i_track],
#                    result_list[i_track]],
#                fmt="%12.8f %12.8f %12.8f %8d %12.8f %8d")

print("Have a nice day, Good bye!")
