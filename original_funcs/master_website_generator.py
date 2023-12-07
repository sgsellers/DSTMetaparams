"""
Created 2023-06-09 Sean G. Sellers
Given a path to an observing date in the archive, creates a nice landing page with metaparam plot.
Calls other functions defined in this directory.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help='Path to date directory')
parser.add_argument("-r", "--rosa_scale", help="ROSA arcsec per pixel (default 0.058)")
parser.add_argument("-z", "--zyla_scale", help="Zyla arcsec per pixel (default 0.0845)")
parser.add_argument("-f", "--zyla_fps", help="Zyla frames per second (default none, program will try to find automatically")
parser.add_argument("-d", "--zyla_shape", help="Zyla data shape if other than (2048,2048). Format x,y")
parser.add_argument("-p", "--p", help="Force redo pointing plot", action='store_true')
parser.add_argument("-t","--t",help="Force redo timing plot", action='store_true')
parser.add_argument("-s", "--skip_seeing", help="Skip creation of seeing plots", action="store_true")
args = parser.parse_args()

indir = args.input

if args.rosa_scale:
    rosa_dxy = args.rosa_scale
else:
    rosa_dxy = 0.058

if args.zyla_scale:
    zyla_dxy = args.zyla_scale
else:
    zyla_dxy = 0.0845

if args.zyla_fps:
    zyla_fps = args.zyla_fps
else:
    zyla_fps = None

if args.zyla_shape:
    zyla_shape=(int(args.zyla_shape.split(",")[0]),int(args.zyla_shape.split(",")[1]))
else:
    zyla_shape=None

# Step 1: Create Seeing plots (takes a while)
# Step 2: Create Observing time plot
# Step 3: Create Pointing plot (uses txt file generated in Step 2)
# Step 4: Generate webpage
# Step 5: Update Monthly Overview

import create_seeing_plot, create_timing_plot, create_pointing_plot, generate_webpage, generate_month,data_quality
import os

if not args.skip_seeing:
    create_seeing_plot.automatic(indir, zyla_shape=zyla_shape)
#try:
if args.t:
    create_timing_plot.automatic(indir, zyla_fps, force_redo='y')
else:
    create_timing_plot.automatic(indir, zyla_fps)
#except:
#    print("Timing plot could not be created. Proceeding...")
#try:
if args.p:
    create_pointing_plot.automatic(indir, rosa_dxy, zyla_dxy, force_redo="y")
else:
    create_pointing_plot.automatic(indir, rosa_dxy, zyla_dxy)
data_quality.auto_update_dq_files(indir)
#except:
#    print("Context image could not be created. Proceeding...")
generate_webpage.automatic(indir)
if indir[-1] == "/":
    indir = indir[:-1]
monthstr = os.path.split(indir)[0]
print(monthstr)
generate_month.automatic(monthstr)
