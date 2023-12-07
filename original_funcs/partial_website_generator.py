import create_timing_plot, create_pointing_plot, generate_webpage, generate_month, create_seeing_plot
import os
import glob
import tqdm

master_dirs = sorted(glob.glob("/sunspot/solardata/2*"))

date_dirs = []
for year in master_dirs:
    for month in glob.glob(year + "/*/"):
        for day in glob.glob(month + "*/"):
            if "RECYCLE" not in day:
                date_dirs.append(day)

# Kludge cause everything after this date has a page already....
date_dirs = sorted(date_dirs,reverse=True)
# date_dirs = sorted(date_dirs[:210-45-100-42-9],reverse=True)
f = open("no_seeing_plot_made.txt","w")
g = open("no_timing_plot_made.txt","w")
for i in tqdm.tqdm(range(len(date_dirs))):
    print(date_dirs[i])
    try:
        create_seeing_plot.automatic(date_dirs[i])
    except:
        print("No seeing plot available for", date_dirs[i])
        print("Writing date to no_seeing_plot_made.txt")
        f.write(date_dirs[i] + "\n")
    try:
        create_timing_plot.automatic(date_dirs[i], None, force_redo='n')
    except:
        print("No Timing plot available for", date_dirs[i])
        print("Writing date to no_timing_plot_made.txt")
        g.write(date_dirs[i] + "\n")
    try:
        create_pointing_plot.automatic(date_dirs[i], 0.058, 0.0845)
    except:
        print("No pointing plot for ",date_dirs[i])
    generate_webpage.automatic(date_dirs[i])
    monthstr = os.path.split(os.path.split(date_dirs[i])[0])[0]
    generate_month.automatic(monthstr)
f.close()
g.close()
