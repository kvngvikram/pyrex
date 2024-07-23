


# Crossover extraction using PyReX

- Along track data with three columns (lon, lat, z) in ASCII format (`.txt`, `.dat`) should be placed in `input` folder.
- The output of PyReX will be saved in the folder `output_detection_files`. Make a folder if it doesn't exist.
- Config script `config_cross.py` along with `detection.py`, `correction.py`, `stacking.py` and a module file `cross_over_module.py` are placed in `codes` folder.

To run the detection:

1) Go to codes folder
	```
	$ cd codes
	```
2) Users can modify the `config_cross.py` as per the requirement. The `config_cross.py` script is already set to run detection from files in input folder. 

3) `config_cross.py` has sections with `pyrex_group_a_detect` and `pyrex_group_b_detect` as `calc_id`.

4) Run the detection for *Group A* with its corresponding `calc_id`
	```
	$ python detection.py pyrex_group_a_detect
	```

5) Similarly, run the detection for *Group B* with its corresponding `calc_id`
	```
	$ python detection.py pyrex_group_b_detect
	```

6) Extracted crossovers are present in `output_detection_files` folder and named corresponding to their `calc_id`.


## Auxillary codes

Similarly, `correction.py` and `stacking.py` can be run by adding corresponding sections in `config_cross.py` as 
```
$ python stacking.py stacking_calc_id
```
for stacking and
```
$ python correction.py corrections_calc_id
```
for corrections.

