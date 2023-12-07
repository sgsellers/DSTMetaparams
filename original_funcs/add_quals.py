import glob
import os
import generate_webpage as gw
import generate_month as gm
import tqdm

all_dirs = sorted(glob.glob("/sunspot/solardata/20*/*/*/"), reverse=True)[45:-1]
all_months = sorted(glob.glob("/sunspot/solardata/20*/*/"), reverse=True)

#for i in tqdm.tqdm(range(len(all_dirs)), desc="Updating daily overviews"):
#    gw.automatic(all_dirs[i])

for i in tqdm.tqdm(range(len(all_months)), desc="Updating monthly overviews"):
    gm.automatic(all_months[i])
