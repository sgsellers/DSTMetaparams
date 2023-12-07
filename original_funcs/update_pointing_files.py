import update_pointing_funcs
import os
import glob
import tqdm

all_dirs = sorted(glob.glob("/sunspot/solardata/20*/*/*/"), reverse=True)[:-1]
for i in tqdm.tqdm(range(len(all_dirs))):
    update_pointing_funcs.automatic(all_dirs[i])
