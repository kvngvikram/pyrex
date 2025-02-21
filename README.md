PyReX
=====
A recursion based crossover detection algorithm in Python for along-track geophysical measurements.

Refer to the [paper](https://doi.org/10.1029/2024EA003932) for a detailed description of the algorithm used and
a basic workflow of implementation of the codes.


## Steps to run a calculation:
- Place the required files in the working directory, i.e.
    - script to run : *detection.py*, *correction.py* and/or *stacking.py*. **These need not be edited by the user**
    - *cross_over_module.py* file which contains necessary functions used by other scripts and **need not be edited by user**
    - *config_cross.py* file, to be **edited by user** to set up the calculation.
    - Necessary data files to work with (in ASCII formats).
- Edit *config_cross.py* by adding your own section with a unique *calc_id* or modify existing ones as per your needs.
- Run the calculation using the command `$ python script.py calc_id`

You can get these from the *codes* directory

Each calculation/run has an associated *calc_id*.
The settings/configuration of any calculation (by any of the script) has to be done by the user inside
*config_cross.py* file. This file can have one or multiple sections, each with a different *calc_id* with corresponding
configuration for that particular calculation.

**Please see the example/reference *config_cross.py* in the *codes* folder for details and explanations of each option**

Feel free to raise any issue, and we will try to address them as much as we can.
Info on any minor updates or bug fixes will be informed here.

#  How to cite
Please cite us as follows if you have used PyReX in your studies.

 *Vikram, K. V. N. G., Krishna, D. V. P., & Sreejith, K. M. (2025).
 PyReX: A recursion based crossover detection algorithm in Python for along-track geophysical measurements.
 Earth and Space Science, 12, e2024EA003932. [https://doi.org/10.1029/2024EA003932](https://doi.org/10.1029/2024EA003932)*

If you want to replicate the entire study of the paper, please contact us, and
we will be happy to share the full satellite altimetry data we used (~6 GB).

### Requirements

- numpy
- pandas
- numba
- scipy

These probably are already present in a *base* anaconda installation.

## Running the test data

Two datasets (*Group A* and *Group B*) used in the [paper](https://doi.org/10.1029/2024EA003932) for testing crossover detection between the tracks of
ship borne free-air gravity measurements from NGDC is placed in the *input_data/group_a* and *input_data/group_b*
folders. The example *config_cross.py* in *codes* folder is preconfigured with two *calc_id*s namely
*pyrex_group_a_detect* and *pyrex_group_b_detect* for crossover detection of corresponding datasets.

- Go to codes folder
```
$ cd codes
```
- Run the detection for *Group A* with its corresponding `calc_id`
```
$ python detection.py pyrex_group_a_detect
```

- Similarly, run the detection for *Group B* with its corresponding `calc_id`
```
$ python detection.py pyrex_group_b_detect
```

- Extracted crossovers are present in *output_detection_files* folder and named corresponding to their *calc_id*.
