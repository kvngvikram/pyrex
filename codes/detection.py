#!/usr/bin/env python
from multiprocessing import Pool
# import multiprocessing as mp
import numpy as np
from numba import set_num_threads, jit
import pandas as pd
import glob
import os
import sys
import re
sys.path.insert(0, os.getcwd())
from cross_over_module import (cross_distances, get_break_indices, lon_shift,
                               distance_on_track, get_pseudo_segment_flags)  # noqa
from config_cross import setup, calc  # noqa

# TODO: remove precalculate flags and precalculation sections later
# because we have proved that precalculation is better with speed
# But not sure, since precalculation takes more memory ?

calc_id = sys.argv[1]
calc_obj = calc(calc_id)

data_glob_string_list = calc_obj.data_glob_string_list
detection_save_filename = calc_obj.detection_save_filename


lon_shift_condition_value = (setup().lon_shift_condition_value
                             if hasattr(setup, "lon_shift_condition_value")
                             else 180)


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


if hasattr(calc_obj, "data_columnn_numbers"):
    data_columnn_numbers = calc_obj.data_columnn_numbers
elif hasattr(setup(), "data_columnn_numbers"):
    data_columnn_numbers = setup().data_columnn_numbers
else:
    data_columnn_numbers = [0, 1, 2]
print(f"data_columnn_numbers: {data_columnn_numbers}")


detection_chunksize = (calc_obj.detection_chunksize
                       if hasattr(calc_obj, "detection_chunksize")
                       else 1)

detection_intersection_type = (calc_obj.detection_intersection_type
                               if hasattr(calc_obj,
                                          "detection_intersection_type")
                               else "line")


# only "simple" for now. Piece wise linear interpolation for Z values
detection_interpolation_type = (calc_obj.detection_interpolation_type
                                if hasattr(calc_obj,
                                           "detection_interpolation_type")
                                else "simple")


detection_type = (calc_obj.detection_type  # "internal" "external" "both"
                  if hasattr(calc_obj, "detection_type")
                  else "both")

filename_match_pattern_self = (calc_obj.filename_match_pattern_self
                               if hasattr(calc_obj,
                                          "filename_match_pattern_self")
                               else "")

filename_match_pattern_other = (calc_obj.filename_match_pattern_other
                                if hasattr(calc_obj,
                                           "filename_match_pattern_other")
                                else "")

recurse_level_limit = (calc_obj.recurse_level_limit
                       if hasattr(calc_obj, "recurse_level_limit")
                       else 50)

rn = (calc_obj.rn
      if hasattr(calc_obj, "rn")
      else None)

precalculate_break_indices = (calc_obj.precalculate_break_indices
                              if hasattr(calc_obj,
                                         "precalculate_break_indices")
                              else True)

precalculate_track_distances = (calc_obj.precalculate_track_distances
                                if hasattr(calc_obj,
                                           "precalculate_track_distances")
                                else True)

precalculate_pseudo_segment_flags = (
        calc_obj.precalculate_pseudo_segment_flags
        if hasattr(calc_obj, "precalculate_pseudo_segment_flags") else True)

precalculate_min_max = (
        calc_obj.precalculate_min_max
        if hasattr(calc_obj, "precalculate_min_max") else True)

pre_read_data_files = (calc_obj.pre_read_data_files
                       if hasattr(calc_obj, "pre_read_data_files")
                       else True)


set_num_threads(number_of_numba_threads)


track_file_names = []
for glob_element in data_glob_string_list:
    track_file_names += sorted(glob.glob(glob_element))  # append list to list

track_file_names = np.asarray(track_file_names)

n_tracks = track_file_names.size  # number of tracks
print(f"Number of tracks in total: {n_tracks}, starting to read")


def file_read(name):
    print(f"\rreading {name}                            ", end='')
    data_i = np.asarray(
            pd.read_csv(name, header=None, sep=r"\s+", dtype=float)
            )[:, data_columnn_numbers]
    data_i[:, 0] = lon_shift(data_i[:, 0])
    data_i = np.c_[data_i[:, 0], data_i[:, 1], data_i[:, 2]]

    return data_i


if pre_read_data_files:
    print("created a Pool of processes, for reading data")

    with Pool(processes=number_of_processes) as P:
        data = P.map(file_read, track_file_names)

    print("\rclosing data reading pool                                       ")
else:
    print('Track data files will be read as and when required.')
    print('This means same file is read multiple times !!')
    data = None
    precalculate_break_indices = False
    precalculate_track_distances = False


if precalculate_break_indices:
    def wrapper_for_get_break_indices(i, data=data):
        return get_break_indices(data[i][:, 0], data[i][:, 1])

    print("created a Pool of getting breaking indices")

    with Pool(processes=number_of_processes) as P:
        break_indices = P.map(wrapper_for_get_break_indices, range(n_tracks))

    print("closing break indices calculation pool")
else:
    break_indices = None


if precalculate_track_distances:
    def wrapper_for_get_track_distances(i, data=data):
        return distance_on_track(data[i][:, 0], data[i][:, 1])

    print("created a Pool of getting track_distances")

    with Pool(processes=number_of_processes) as P:
        track_distances = P.map(wrapper_for_get_track_distances,
                                range(n_tracks))

    print("closing track_distances calculation pool")
else:
    track_distances = None


if precalculate_pseudo_segment_flags:
    def wrapper_for_get_pseudo_segment_flags(i, data=data):
        return get_pseudo_segment_flags(data[i][:, 0], data[i][:, 1])

    print("created a Pool of getting Pseudo segment flags")

    with Pool(processes=number_of_processes) as P:
        ps_flags = P.map(wrapper_for_get_pseudo_segment_flags, range(n_tracks))

    print("closing Pseudo segment flags calculation pool")
else:
    ps_flags = None


if precalculate_min_max:

    @jit(nopython=True)
    def get_min_max(x, y):
        return np.array([np.min(x), np.max(x), np.min(y), np.max(y)])

    def wrapper_for_get_min_max(i, data=data):
        return get_min_max(data[i][:, 0], data[i][:, 1])

    print("created a Pool of getting min max values")

    with Pool(processes=number_of_processes) as P:
        min_max_list = P.map(wrapper_for_get_min_max, range(n_tracks))

    print("closing min max calculation pool")
else:
    min_max_list = None


# these track combinations will be forwarded to detection
@jit(nopython=True)
def combination_generator(n_tracks):  # needs number of tracks

    number_of_combinations = int(n_tracks*(n_tracks - 1)/2 + n_tracks)
    count = 0
    i = 0
    j = 0
    while count < number_of_combinations:
        yield [i, j]

        count += 1
        if j == i:
            i += 1
            j = 0
        else:
            j += 1


self_re = re.compile(filename_match_pattern_self)
other_re = re.compile(filename_match_pattern_other)


# @njit(cache=True)
def single_calc(ij,
                data=data,
                break_indices=break_indices,
                track_distances=track_distances,
                ps_flags=ps_flags,
                min_max_list=min_max_list,
                track_file_names=track_file_names,
                detection_type=detection_type,
                self_re=self_re,
                other_re=other_re
                ):
    i = ij[0]
    j = ij[1]

    names = track_file_names
    allow_filename_flag = (
                (self_re.search(names[i]) and other_re.search(names[j])) or
                (self_re.search(names[j]) and other_re.search(names[i])))
    if (
            (detection_type == 'internal' and i != j) or
            (detection_type == 'external' and i == j) or
            (not allow_filename_flag)
            ):
        return None

    if data is None:  # can use global pre_read_data_files
        data_i = file_read(names[i])
        data_j = file_read(names[j])
    else:
        data_i = data[i]
        data_j = data[j]

    if break_indices is None:  # can use global precalculate_break_indices
        bi_i = None
        bi_j = None
    else:
        bi_i = break_indices[i]
        bi_j = break_indices[j]

    if track_distances is None:  # can use global precalculate_track_distances
        d_i = None
        d_j = None
    else:
        d_i = track_distances[i]
        d_j = track_distances[j]

    if ps_flags is None:  # can use global precalculate_pseudo_segment_flags
        ps_f_i = None
        ps_f_j = None
    else:
        ps_f_i = ps_flags[i]
        ps_f_j = ps_flags[j]

    if min_max_list is None:  # can use global precalculate_min_max
        min_xi = -np.inf
        max_xi = np.inf
        min_yi = -np.inf
        max_yi = np.inf
        min_xj = -np.inf
        max_xj = np.inf
        min_yj = -np.inf
        max_yj = np.inf
    else:
        min_xi = min_max_list[i][0]
        max_xi = min_max_list[i][1]
        min_yi = min_max_list[i][2]
        max_yi = min_max_list[i][3]
        min_xj = min_max_list[j][0]
        max_xj = min_max_list[j][1]
        min_yj = min_max_list[j][2]
        max_yj = min_max_list[j][3]

    lon1 = data_i[:, 0]
    lat1 = data_i[:, 1]
    z1 = data_i[:, 2]
    bi_1 = bi_i
    d_1 = d_i
    ps_f_1 = ps_f_i
    lon2 = data_j[:, 0]
    lat2 = data_j[:, 1]
    z2 = data_j[:, 2]
    bi_2 = bi_j
    d_2 = d_j
    ps_f_2 = ps_f_j

    min_x1 = min_xi
    max_x1 = max_xi
    min_y1 = min_yi
    max_y1 = max_yi
    min_x2 = min_xj
    max_x2 = max_xj
    min_y2 = min_yj
    max_y2 = max_yj

    intersection_type = detection_intersection_type
    interpolation_type = detection_interpolation_type
    internal_flag = i == j

    print(f'\r{i, j}   \t -> doing : {names[i]}          {names[j]}                            ',  # noqa
          end='')

    (n_int, x_int, y_int,
     ls1_idx, z1_int, d1,
     ls2_idx, z2_int, d2) = cross_distances(
             lon1, lat1, z1, lon2, lat2, z2,
             bi_1=bi_1, bi_2=bi_2,
             d_1=d_1, d_2=d_2,
             ps_f_1=ps_f_1, ps_f_2=ps_f_2,
             rn=rn,
             min_x1=min_x1, max_x1=max_x1,
             min_y1=min_y1, max_y1=max_y1,
             min_x2=min_x2, max_x2=max_x2,
             min_y2=min_y2, max_y2=max_y2,
             recurse_level_limit=recurse_level_limit,
             intersection_type=intersection_type,
             internal_flag=internal_flag,
             interpolation_type=interpolation_type)

    a_track1_idx = np.zeros(n_int, dtype=int) + int(i)
    a_track2_idx = np.zeros(n_int, dtype=int) + int(j)

    print(f'\r{i, j}   \t -> done  : {names[i]}          {names[j]}                            ',  # noqa
          end='')
    # for n in range(n_int):
    #     print(f'\r{i}\t{j}                   ', end='')

    out = np.c_[a_track1_idx, a_track2_idx, x_int, y_int, ls1_idx, z1_int,
                d1, ls2_idx, z2_int, d2]

    if n_int > 0:
        return out
    else:
        return None


def wrapper_single_calc(i):
    # initialising
    j = 0

    out = np.c_[  # dummy value for getting the right array shapes
            np.array([-2]), np.array([-2]),
            np.array([np.nan]), np.array([np.nan]),
            np.array([-2]), np.array([np.nan]), np.array([np.nan]),
            np.array([-2]), np.array([np.nan]), np.array([np.nan])]

    while j <= i:
        return_result = single_calc([i, j])
        if return_result is not None:
            out = np.concatenate((out, return_result))
        j += 1

    if out.shape[0] > 1:
        return out[1:]
    else:
        return


# --
P = Pool(processes=number_of_processes)
print("created a Pool of processes, doing calculations now")


print('detected crossover between : ')
result_generator = P.imap(wrapper_single_calc, range(n_tracks),
                          chunksize=detection_chunksize)

result = (i for i in result_generator if i is not None)
# using () instead of [] is more memory efficient ?
# see designcise.com/web/tutorial/how-to-remove-all-none-values-from-a-python-list#using-generator-expression   # noqa
result = np.concatenate(list(result))

P.close()
P.join()

print("\rcalculations complete!                                            ")
print("Saving data now")
# --


result_df = pd.DataFrame({"Track1": np.take(track_file_names,
                                            result[:, 0].astype(int)),
                          "Track2": np.take(track_file_names,
                                            result[:, 1].astype(int)),
                          "x_int": result[:, 2],
                          "y_int": result[:, 3],
                          "ls_index_1": result[:, 4].astype(int),
                          "z1": result[:, 5],
                          "d1": result[:, 6],
                          "ls_index_2": result[:, 7].astype(int),
                          "z2": result[:, 8],
                          "d2": result[:, 9]
                          })

result_df = result_df.astype({"Track1":       "str",
                              "Track2":       "str",
                              "x_int":      "float",
                              "y_int":      "float",
                              "ls_index_1":   "int",
                              "z1":         "float",
                              "d1":         "float",
                              "ls_index_2":   "int",
                              "z2":         "float",
                              "d2":         "float"})


result_df.to_csv(detection_save_filename,
                 sep="\t", na_rep="NaN", index=False, header=True)
