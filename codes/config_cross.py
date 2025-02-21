#!/usr/bin/env python
# author: Vikram KVNG
# email: kvngvikram@sac.gov.in kvngvikram@gmail.com
# date of creation: 23-07-2024
# Example config_cross.py

# import thise when necessary
import numpy as np
# import glob
# import re
# import subprocess

# Go through the actual script to understand how each of these options are used

# Things to be familier with: f-string, glob, classes (priliminary) ..

# make folders like ./detection_files and ./correction_info beforehand


class setup:  # class independent of calc_id. Common options for all.
    def __init__(self):

        self.semi_major_axis = 6378.1363 * 1000  # m
        # (semi_major-semi_minor)/semi_minor
        self.flattening_coefficient = 1/298.257

        # OPTIONAL configuration and their default values
        # These can also be set in calc_id sections. The following will
        # be used if they are exclusively not set in calc_id section
        # These can be set here once as common ones instead of setting multiple
        # times in calc_id sections
        self.number_of_processes = 1
        self.data_columnn_numbers = [0, 1, 2]  # for Longitude, latitude, Z
        self.input_data_delimiter = r'\s+'  # any number of whitespaces


class calc:
    def __init__(self, calc_id):

        self.calc_id = calc_id

        if (calc_id == 'detect'):  # for detection.py

            print(f"Calc id is {calc_id}")

            # input tracks for detection
            self.data_glob_string_list = [
                    "./gmt/ngdc_B/*.m77t.xyz",
                    # "./gmt_scaling/data/*",
                    "dummydummydummy/*"]  # don't forget the star

            # output detection file
            self.detection_save_filename = f"./detection_files/detection_{calc_id}.txt"  # noqa

            ##############################################
            # OPTIONAL settings with their default values
            self.number_of_processes = 1
            self.detection_type = 'both'  # or 'external' or 'internal' Xovers
            self.detection_intersection_type = 'line'  # 'line' or 'arc'
            # for lon, lat and z in track files
            self.data_columnn_numbers = [0, 1, 2]
            self.input_data_delimiter = r'\s+'  # any number of whitespaces

            # What if you want to find detections only between two subsets of
            # tracks from all the input tracks.
            # You can select the subsets by matching the substrings from
            # filenames with a regex pattern.
            # regex pattern for matching by re.search()
            self.filename_match_pattern_self = ''
            self.filename_match_pattern_other = ''
            # for example: only find detections between jason1 and jason2 only
            # but not between jason1 and jason1 or jason2 and jason2, then
            # use this option. This is different from 'internal' and 'external'
            # crossovers.
            # If the filenames are js1_cXXX_tXXX.dat and js2_cXXX_tXXX.dat
            # then set filename_match_pattern_self = 'js1' and
            # filename_match_pattern_other = 'js2'
            # For selecting jason2 and jason3 together but jason1 seperately
            # you can use 'js1' and 'js2|js3' (check regex syntax of re.search)
            # NOTE: here only in detection swapping 'self' and 'other' has no
            # difference. Not the case for corrections

        elif (calc_id == 'stack'):  # for stacking.py

            # Stacking can only be done for same kind of ERM mesh/orbit.
            # For example ERM tracks of Topex, and Jason series can be stacked,
            # but they cannot be stacked together with the Jason interleaved
            # tracks.

            # Similarly Tracks of SARAL, Envisat and ERS1 and ERS2 can be
            # stacked because they are in the same orbit

            # A seperate example is given after this section named
            # 'stack_example' for more clarity

            # the idea is to give a list of different tracks (same orbit) and
            # select the longest track (in any cycle) as base track and select
            # all the data of same track (different cycles) for averaging at
            # the locations of base track's points.
            # NOTE: The longest track means the one with largest number of
            # data points assuming regular sampling frequency.

            print(f"Calc id is {calc_id}")

            self.number_of_tracks = 1002  # for SARAL AltiKa
            self.cutoff = 3500  # meters the radius for averaging

            self.stacking_file_read_glob_f_string_list = [
                    './folder_path_saral_erm/saral_erm_*_{i_track+1:04}.dat',  # SARAL ERM tracks (1002 tracks) of any number of cycles  # noqa
                    './folder_path_ers1_erm/ers1_erm_*_{i_track+1:04}.dat',    # Same as SARAL but for ERS1  # noqa
                    './folder_path_ers2_erm/ers2_erm_*_{i_track+1:04}.dat',    # Same as SARAL but for ERS2  # noqa
                    './folder_path_envisat_erm/env_erm_*_{i_track+1:04}.dat',  # Same as SARAL but for Envisat  # noqa
                    './folder_path_optional_base_tracks/base_*_{i_track+1:04}.dat',  # OPTIONAL: If you have synthetic tracks to use as base  # noqa
                    ]
            # This is a list of strings. Each string will be first evaluated
            # as an f-string and then given to glob to get the track files.
            # The '*' is for glob matching the file name.
            # And the contentes inside '{}' will be evaluated for f-string.
            # 'i_track' is the name of the variable used inside the stacking
            # code for track number (index).
            # And it starts with '0' while the file names will start with '1'.
            # So the '+1' is used. The ':04' is for the number of characters in
            # filenames for track number, i.e. '0001', '0002' ... '1002'

            # f-string to save the stacked tracks
            self.stacking_save_filepaths_f_string = './save_folder/erm_stack_{i_track+1:04}.dat'  # noqa
            # create the save_folder beforehand

            ##############################################
            # OPTIONAL settings with their default values
            self.number_of_processes = 1
            # maximum iterations for removing outliers while averaging
            self.max_iter = 20
            # integer: consecutive data point to skip when considering the
            # base track for example set it as 2 to consider every alternate
            # point in base track
            self.data_skip_value = 1
            self.input_data_delimiter = r'\s+'  # any number of whitespaces

            # now the idea is:
            # give all the tracks in 'stacking_file_read_glob_f_string_list'
            # option. But have the ability to select few subset of these tracks
            # for finding the base tracks (longest in this subset). And have
            # the ability to select same/different subset of tracks whose
            # Z-values are actually used for averaging. This is possible useing
            # the following two variables which select by matching the
            # filenames based on a parrern (example: 'srl' in filename).
            # These are regex match patterns used in
            # pandas_series.str.contains(pattern)
            self.longest_track_name_match_pattern = ''  # disabled. By matching everything  # noqa
            self.tracks_for_averaging_name_match_pattern = ''  # disabled. By matching everything  # noqa

        elif (calc_id == 'stacking_example'):

            # Goal: Stack all the ERM data of SARAL.
            # Data available: SARAL ERM, Envisat ERM, ERS1 ERM and ERS2 ERM

            # Hypothetical Issue: SARAL has location errors and accross-track
            # drift in some orbits but the Z values (SSH) are accurate.

            # So get the base tracks from Envisat, ERS1, and ERS2 and use
            # Z values (SSH) from SARAL

            self.number_of_tracks = 1002

            # Give all the ERM data from four satellites, all cycles
            self.stacking_file_read_glob_f_string_list = [
                    "./saral/srl_erm_C*_P{i_track+1:04}.dat",
                    "./ers1/er1_erm_C*_P{i_track+1:04}.dat",
                    "./ers2/er2_erm_C*_P{i_track+1:04}.dat",
                    "./envisat/env_erm_C*_P{i_track+1:04}.dat"
                    ]
            # File name examples are like
            # ./saral/srl_erm_Cycle1_P0001.dat
            # ./saral/srl_erm_Cycle20_P0100.dat
            # ./saral/srl_erm_Cycle100_P1001.dat
            # ./ers1/er1_erm_Cycle11_P0001.dat
            # ./ers2/er2_erm_Cycle20_P0100.dat
            # ./envisat/env_erm_Cycle100_P1002.dat

            # save the 1002 stacked tracks
            self.stacking_save_filepaths_f_string = (
                    "./stacks/srl_erm_stack_{i_track+1:04d}.dat")

            # regex match pattern to select the subset of tracks
            # based on filenames !!
            # The longest among this subset will be used as base track
            self.longest_track_name_match_pattern = "env|er1|er2"
            # Use Z-values from SARAL
            self.tracks_for_averaging_name_match_pattern = "srl"

            # Take every 3rd point of the selected base track for output.
            # This is basically reducing the sampling rate
            self.data_skip_value = 3

            self.cutoff = 3500  # 3.5km radius for 1Hz data

        elif (calc_id == 'correction'):

            # read the crossovers from detection file, select the subset of
            # tracks to correct i.e. 'self'. Correction is done by trying to
            # stick them to another subset 'other'. Selection of these subsets
            # is done by matching the filenames from detection file.
            # Once corrected, update and save the new detection file, save the
            # corrected tracks in a new folder and save a corrections file
            # has info on how each track is corrected (fit parameters)

            self.detection_input_filename = './detection_files/detection_calc_id.txt'  # noqa
            self.post_correction_detection_filename = f'./detection_files/detection_{calc_id}.txt'  # noqa

            # folder location to save the corrected tracsk
            self.corrected_folder = f'./{calc_id}_corrected_tracks'

            self.correction_save_filename = f'./correction_files/corrections_{calc_id}.txt'  # noqa

            # You can select the subsets by matching the substrings from
            # filenames with a regex pattern.
            # regex pattern for matching by re.search()
            self.filename_match_pattern_self = 'gm'  # filenames contain gm
            self.filename_match_pattern_other = 'erm'  # filenames contain erm
            self.correction_weight = 1  # use 0.5 for same self and other sets
            # can select different types of datasets like 'saral_erm|jason_erm'
            # if you have saved the track filenames that way.

            self.fundamental_distance = 20_000_000.0  # 20,000 km as example
            self.fit_order = 3  # integer. -1, 0, 1, 2, 3, ...

            ##############################################
            # OPTIONAL settings with their default values
            self.fit_margin = 0  # integer
            # 0 means fit will be attempted if #observations (valid crossovers)
            # is equal or higher than #unknowns to fit
            # if a margin is given then #observations should be atleast that
            # margin amount higher than #unknowns. Or else fit will not be done

            # for not considering the outliers in the crossover differences
            # when fitting by giving thresholds. By default all the crossover
            # differences will be used.
            self.xdiff_min = -np.inf
            self.xdiff_max = np.inf

            # if distance between two crossovers is two high then interpolate
            # the fit value from the fit values at the crossover points
            # This will be done if this distance is greater than this threshold
            self.interpolation_threshold = np.inf

            # for the files named in the detection_input_filename
            self.data_columnn_numbers = [0, 1, 2]
            self.input_data_delimiter = r'\s+'  # any number of whitespaces

            self.curve_fit_maxfev = None  # integer
            # to be used in scipy.optimize.curve_fit function

            self.number_of_processes = 1

        #######################################################################
        #          The following two are for the provided test data
        #######################################################################
        elif (calc_id == "pyrex_group_a_detect"):

            print(f"Calc id is {calc_id}")

            self.data_glob_string_list = [
                    "../input_data/group_a/*.m77t.xyz",
                    # "./gmt_scaling/data/*",
                    "dummydummydummy/*"]
            # don't forget the star

            self.detection_save_filename = f"../output_detection_files/detection_{calc_id}.txt"  # noqa

            self.detection_type = 'external'

        elif (calc_id == "pyrex_group_b_detect"):

            print(f"Calc id is {calc_id}")

            self.data_glob_string_list = [
                    "../input_data/group_b/*.m77t.xyz",
                    # "./gmt_scaling/data/*",
                    "dummydummydummy/*"]
            # don't forget the star

            self.detection_save_filename = f"../output_detection_files/detection_{calc_id}.txt"  # noqa

            self.detection_type = 'external'
