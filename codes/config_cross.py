class setup:
    def __init__(self):
        self.number_of_numba_threads = 1
        self.number_of_processes = 1

        self.semi_major_axis = 6378.1363 * 1000  # m
        # (semi_major-semi_minor)/semi_minor
        self.flattening_coefficient = 1/298.257

        self.data_columnn_numbers = [0, 1, 2]


class calc:
    def __init__(self, calc_id):

        self.calc_id = calc_id

        if (calc_id == "pyrex_group_a_detect"):

            print(f"Calc id is {calc_id}")

            self.data_glob_string_list = [
                    "../input/group_a/*.m77t.xyz",
                    # "./gmt_scaling/data/*",
                    "dummydummydummy/*"]
            # don't forget the star

            self.detection_save_filename = f"../output_detection_files/detection_{calc_id}.txt"  # noqa

            self.detection_type = 'external'

        elif (calc_id == "pyrex_group_b_detect"):

            print(f"Calc id is {calc_id}")

            self.data_glob_string_list = [
                    "../input/group_b/*.m77t.xyz",
                    # "./gmt_scaling/data/*",
                    "dummydummydummy/*"]
            # don't forget the star

            self.detection_save_filename = f"../output_detection_files/detection_{calc_id}.txt"  # noqa

            self.detection_type = 'external'
