import numpy as np, matplotlib.pyplot as plt, sunpy.map as smap, matplotlib, astropy.units as u, glob, astropy.io.fits as fits, matplotlib.dates as mdates
from sunpy.net import Fido, attrs as a
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy import timeseries as ts
import os

def create_firs_obssum(firs_level0_dir,date):
    """Finds FIRS level-0 files, sorts them by type (sflat, polcal, data, dark, lampflat, and various target/linegrid slide observations) and returns starttimes, endtimes, and types.
    """
    list_of_firs_files = sorted(glob.glob(firs_level0_dir + 'firs.2.*'))
    list_of_obs = []
    for i in range(len(list_of_firs_files)):
        list_of_obs.append('.'.join(list_of_firs_files[i].split('.')[:-2]))
    list_of_obs = list(set(list_of_obs))
    starttime = []
    endtime = []
    obstype = []
    slat = []
    slng = []
    mean_see = []
    std_see = []
    for i in range(len(list_of_obs)):
        files_in_series = sorted(glob.glob(list_of_obs[i]+'*'))
        f0 = fits.open(files_in_series[0])
        f1 = fits.open(files_in_series[-1])
        try:
            otype = f0[0].header['COMMENT'][0]
            if f0[0].header['COMMENT'][0] == 'pcal':
                st = np.datetime64(f0[0].header['OBS_STAR'])
                if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):
                    obstype.append(otype)
                    starttime.append(np.datetime64(f0[0].header['OBS_STAR']))
                    endtime.append(np.datetime64(f1[0].header['OBS_END']))
                    slat.append(str(f0[0].header['DST_SLAT']))
                    slng.append(str(f0[0].header['DST_SLNG']))
                    see = []
                    for file in files_in_series:
                        try:
                            f = fits.open(file)
                            scmon = f[0].header['DST_SEE']
                            see.append(scmon)
                        except:
                            pass
                    mean_see.append(np.nanmean(see))
                    std_see.append(np.nanstd(see))

            else:
                st = np.datetime64(f0[0].header['OBS_STAR'])
                if (st < (np.datetime64(date) + np.timedelta64(1,"D"))) and (st > np.datetime64(date)):
                    obstype.append(otype)
                    starttime.append(np.datetime64(f0[0].header['OBS_STAR']))
                    endtime.append(np.datetime64(f1[0].header['OBS_END']))
                    if "DST_SLAT" in list(f0[0].header.keys()):
                        slat.append(str(f0[0].header['DST_SLAT']))
                        slng.append(str(f0[0].header['DST_SLNG']))
                    else:
                        slat.append("0")
                        slng.append("0")
                    see = []
                    for file in files_in_series:
                        try:
                            f = fits.open(file)
                            scmon = f[0].header['DST_SEE']
                            see.append(scmon)
                        except:
                            pass
                    mean_see.append(np.nanmean(see))
                    std_see.append(np.nanstd(see))

            f0.close()
            f1.close()
            
        except:
            #### Ran into an issue with untagged file. Skip
            f0.close()
            f1.close()
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slng = np.array(slng)
    mean_see = np.array(mean_see)
    std_see = np.array(std_see)
    firs_obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])

    return firs_obssum


def create_ibis_obssum(ibis_level0_dir, date):
    """Finds IBIS level-0 files, and sorts them by type, returning starttimes, endtimes, and types.
    """

    ibis_subdirectories = sorted(glob.glob(os.path.join(ibis_level0_dir,"*") + "/"))
    starttime = []
    endtime = []
    obstype = []
    slat = []
    slng = []
    mean_see = []
    std_see = []
    for directory in ibis_subdirectories:
        data_dir = glob.glob(directory + "*/")
        for folder in data_dir:
            files = sorted(glob.glob(folder + "*.fits*"))
            if len(files) > 1:
                try:
                    f0 = fits.open(files[0])
                except:
                    f0 = fits.open(files[1])
                starttime.append(np.datetime64(f0[1].header['DATE-OBS']))
                slat.append(str(f0[1].header['DST_SLAT']))
                slng.append(str(f0[1].header['DST_SLNG']))
                f0.close()
                try:
                    f1 = fits.open(files[-1])
                except:
                    f1 = fits.open(files[-2])
                endtime.append(np.datetime64(f1[-1].header['DATE-END']))
                f1.close()
                if "Flat" in directory:
                    obstype.append("sflat")
                elif "Dark" in directory:
                    obstype.append("dark")
                elif "Science" in directory:
                    obstype.append("scan")
                elif "Grid" in directory:
                    obstype.append("lgrd")
                elif "Target" in directory:
                    obstype.append("targ")
                else:
                    obstype.append("othr")
                see = []
                for i in range(len(files)):
                    try:
                        f = fits.open(files[i])
                        for j in range(1, len(f)):
                            try:
                                scmon = f[j].header['DST_SEE']
                            except:
                                scmon = np.nan
                            see.append(scmon)
                    except:
                        pass
                mean_see.append(np.nanmean(see))
                std_see.append(np.nanstd(see))

    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slng = np.array(slng)
    mean_see = np.array(mean_see)
    std_see = np.array(std_see)
    ibis_obssum = np.rec.fromarrays([starttime, endtime, obstype, slat, slng,mean_see,std_see], names=['start','end','type','slat','slng','scin','stdscin'])
    return ibis_obssum


def create_spinor_obssum(spinor_level0_dir,date):
    """Finds SPINOR level-0 files, and sorts the "map" files by type, returning starttimes, endtimes, and types. Based on the way SPINOR work(s)(ed), you should only need to run this on one camera, not all three
    """
    list_of_spinor_files = sorted(glob.glob(spinor_level0_dir + "*.fits*"))
    starttime = []
    endtime = []
    obstype = []
    slat = []
    slng = []
    mean_see = []
    std_see = []
    for i in range(len(list_of_spinor_files)):
        fname = list_of_spinor_files[i]
        sp_file = fits.open(fname)
        # For SPINOR endtimes, we go with the second to last file for the end time.
        # Why? Cause SPINOR crashes...constantly.
        if '.flat.' in fname:
            try:
                ot = 'sflat'
                st = np.datetime64(sp_file[1].header['DATE-OBS'])
                et = np.datetime64(sp_file[-1].header['DATE-OBS'])
                lt = str(sp_file[1].header['DST_SLAT'])
                ln = str(sp_file[1].header['DST_SLNG'])
                if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):

                    obstype.append(ot)
                    starttime.append(st)
                    endtime.append(et)
                    slat.append(lt)
                    slng.append(ln)
                    see = []
                    for i in range(1, len(sp_file)):
                        try:
                            scmon = sp_file[i].header['DST_SEE']
                        except:
                            scmon = np.nan
                        see.append(scmon)
                    mean_see.append(np.nanmean(see))
                    std_see.append(np.nanstd(see))

            except:
                pass
        elif '.lamp.flat.' in fname:
            try:
                ot = 'lflat'
                st = np.datetime64(sp_file[1].header['DATE-OBS'])
                et = np.datetime64(sp_file[-1].header['DATE-OBS'])
                lt = str(sp_file[1].header['DST_SLAT'])
                ln = str(sp_file[1].header['DST_SLNG'])
                if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):

                    obstype.append(ot)
                    starttime.append(st)
                    endtime.append(et)
                    slat.append(lt)
                    slng.append(ln)
                    see = []
                    for i in range(1, len(sp_file)):
                        try:
                            scmon = sp_file[i].header['DST_SEE']
                        except:
                            scmon = np.nan
                        see.append(scmon)
                    mean_see.append(np.nanmean(see))
                    std_see.append(np.nanstd(see))

            except:
                pass

        elif '.cal.' in fname:
            #### Sometimes we have polcals from other days.
            #### Get rid of these.
            try:
                ot = 'pcal'
                st = np.datetime64(sp_file[1].header['DATE-OBS'])
                et = np.datetime64(sp_file[-1].header['DATE-OBS'])
                lt = str(sp_file[1].header['DST_SLAT'])
                ln = str(sp_file[1].header['DST_SLNG'])

                if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):
                    obstype.append(ot)
                    starttime.append(st)
                    endtime.append(et)
                    slat.append(lt)
                    slng.append(ln)
                    see = []
                    for i in range(1, len(sp_file)):
                        try:
                            scmon = sp_file[i].header['DST_SEE']
                        except:
                            scmon = np.nan
                        see.append(scmon)
                    mean_see.append(np.nanmean(see))
                    std_see.append(np.nanstd(see))

            except:
                pass
        else:
            ot = sp_file[1].header['PT4_FS']
            if 'USER1' in ot:
                ot = 'scan'
                
            else:
                ot = ot.lower()
            st = np.datetime64(sp_file[1].header['DATE-OBS'])
            if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):
                obstype.append(ot)
                starttime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                try:
                    endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                except:
                    endtime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))
                see = []
                for i in range(1, len(sp_file)):
                    try:
                        scmon = sp_file[i].header['DST_SEE']
                    except:
                        scmon = np.nan
                    see.append(scmon)
                mean_see.append(np.nanmean(see))
                std_see.append(np.nanstd(see))


    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slng = np.array(slng)
    mean_see = np.array(mean_see)
    std_see = np.array(std_see)
    spinor_obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])
    return spinor_obssum



def create_hsg_obssum(hsg_level0_dir, date):
    """Finds HSG level-0 files, and sorts the "scan" files by type, returning starttimes, endtimes, and types. Based on the way HSG work(s)(ed), you should only need to run this on one camera, not all three
    """
    list_of_hsg_files = sorted(glob.glob(hsg_level0_dir + "*.fits*"))
    starttime = []
    endtime = []
    obstype = []
    slat = []
    slng = []
    mean_see = []
    std_see = []
    for i in range(len(list_of_hsg_files)):
        fname = list_of_hsg_files[i]
        sp_file = fits.open(fname)
        st = np.datetime64(sp_file[0].header['DATE-BGN'])
        if (st < (np.datetime64(date) + np.timedelta64(1,"D"))) and (st > np.datetime64(date)):

            if 'solar_flat' in fname:
                obstype.append('sflat')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'lamp_flat' in fname:
                obstype.append('lflat')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'dark' in fname:
                obstype.append('dark')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'target' in fname:
                obstype.append('target')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'linegrid' in fname:
                obstype.append('line')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'scan' in fname:
                obstype.append('scan')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))
            #### Options for when HSG was run with SPINOR GUI
            elif 'sun.flat' in fname:
                obstype.append('sflat')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'lamp.flat' in fname:
                obstype.append('lflat')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'cal' in fname:
                obstype.append('pcal')
                starttime.append(np.datetime64(sp_file[0].header['DATE-BGN']))
                endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            elif 'map' in fname:
                ot = sp_file[1].header['PT4_FS']
                if 'USER1' in ot:
                    ot = 'scan'
                    
                else:
                    ot = ot.lower()
                obstype.append(ot)
                starttime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                try:
                    endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                except:
                    endtime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))
            else:
                ot = 'othr'
                obstype.append(ot)
                starttime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                try:
                    endtime.append(np.datetime64(sp_file[-1].header['DATE-OBS']))
                except:
                    endtime.append(np.datetime64(sp_file[1].header['DATE-OBS']))
                slat.append(str(sp_file[1].header['DST_SLAT']))
                slng.append(str(sp_file[1].header['DST_SLNG']))

            see = []
            for i in range(1, len(sp_file)):
                try:
                    scmon = sp_file[i].header['DST_SEE']
                except:
                    scmon = np.nan
                see.append(scmon)
            mean_see.append(np.nanmean(see))
            std_see.append(np.nanstd(see))


    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slng = np.array(slng)
    mean_see = np.array(mean_see)
    std_see = np.array(std_see)
    hsg_obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])
    return hsg_obssum


def create_rosa_obssum(rosa_level0_dir,date):
    """Finds ROSA level-0 files, and sorts into flat, dark, and map. There's no way to tell what type of map, so we'll have to do it in post by comparison to FIRS, which maps simultaneously. I'd reccommend feeding it the gband directory. I don't think we ever run ROSA without gband obs."""
    list_of_rosa_files = sorted(glob.glob(rosa_level0_dir + '*.fit*'))
    list_of_obs = []
    for i in range(len(list_of_rosa_files)):
        flist = list_of_rosa_files[i].split('.')
        if len(flist) == 5:
            flist = flist[:-2]
        else:
            flist = flist[:-1]
        obs_str = '.'.join(flist)
        obs_str = obs_str[:-5]
        list_of_obs.append(obs_str)
    list_of_obs = list(set(list_of_obs))
    starttime = []
    endtime = []
    obstype = []
    for i in range(len(list_of_obs)):
        files_in_series = sorted(glob.glob(list_of_obs[i]+'*'))
        f0 = fits.open(files_in_series[0])
        f1 = fits.open(files_in_series[-1])
        st = np.datetime64(f0[1].header['DATE'])
        if (st < (np.datetime64(date) + np.timedelta64(1,'D'))) and (st > np.datetime64(date)):
            starttime.append(np.datetime64(f0[1].header['DATE']))
        ###Since ROSA is more or less killed when we stop observing, sometimes the final extension of the FITS file gets truncated. Given that there's a max of 256 images per file, and they run at ~30 fps, thats 8 seconds missing. We'll (naively) pad it by (len(flist)-1 * f[0].header['HEIRARCH FRAME RATE']) ms
            dframe_ms = int(f1[0].header['HIERARCH FRAME RATE'].split(" ")[0])
            td = np.timedelta64(int((len(f1) - 1) * dframe_ms),'ms')
            endtime.append(np.datetime64(f1[1].header['DATE']) + td)
            if 'flat' in list_of_obs[i]:
                obstype.append("sflat")
            elif 'dark' in list_of_obs[i]:
                obstype.append("dark")
            else:
                obstype.append("map")
        f0.close()
        f1.close()
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)

    rosa_obssum = np.rec.fromarrays([starttime,endtime,obstype],names=['start','end','type'])
    return rosa_obssum

def create_zyla_obssum(zyla_dir, date, zfps=None):
    """My old nemesis. Now we must do the same thing for Zyla, which has no set way of defining the framerate, observation type, or start time outside of the directory names. Generally, we'll want to just pin ROSA and Zyla together, call them one instrument, but filament observations were taken with Zyla only at a weird framerate (64 frames/60 seconds, but at 29 fps). A pain. "O Lethe, enemy and lover, without whom my very existence would be pathetic and vulgar!"
    """
    obs_dir = glob.glob(zyla_dir + "**/DBJ*obs*/",
                        recursive=True)
    obs_dir2 = glob.glob(zyla_dir + "**/DBJ*Obs*/",
                         recursive=True)
    fil_dir = glob.glob(zyla_dir + "**/DBJ*fil*/",
                         recursive=True)
    limb_dir = glob.glob(zyla_dir + "**/DBJ*limb*/",
                         recursive=True)
    ar_dir = glob.glob(zyla_dir + "**/DBJ*_ar*/",
                       recursive=True)
    ar_dir2 = glob.glob(zyla_dir + "**/DBJ*_AR*/",
                        recursive=True)
    bur_dir = glob.glob(zyla_dir + "**/DBJ*burst*/",
                        recursive=True)
    data_dirs = list(
            set(
                obs_dir + 
                obs_dir2 + 
                fil_dir + 
                limb_dir + 
                ar_dir + 
                ar_dir2 + 
                bur_dir
            )
    )
    cal_tags = ['flat','dark','dot','pin','error','line','target','test']
    ct = ['flat','dark','dot','pin','line','target','test']
    all_dirs = glob.glob(
            zyla_dir + "**/DBJ_data*/",
            recursive=True)
    data_dirs2 = []
    cal_dirs = []
    obstype = []
    for h in range(len(all_dirs)):
        if not any(x in all_dirs[h] for x in cal_tags):
            data_dirs2.append(all_dirs[h])
        else:
            for i in range(len(ct)):
                if ct[i] in all_dirs[h]:
                    obstype.append(ct[i])
                    cal_dirs.append(all_dirs[h])
    if len(data_dirs) == 0:
        data_dirs = data_dirs2
    ot = ['map' for x in data_dirs]
    dirs = data_dirs + cal_dirs
    obstype = ot + obstype
    starttime = []
    endtime = []
    for h in range(len(data_dirs)):
        end_folder = data_dirs[h].split("/")[-2]
        folder_tags = end_folder.split("_")[2:]
        timetag = None
        for k in range(len(folder_tags)):
            if len(folder_tags[k]) >= 4:
                if folder_tags[k].isnumeric():
                    timetag = folder_tags[k]
                elif folder_tags[k][:4].isnumeric():
                    timetag = folder_tags[k][:4]
        if timetag:
            if len(timetag)> 4:
                tt_new = timetag[:2] + ':' + timetag[2:4] + ':' + timetag[4:]
            else:
                tt_new = timetag[:2] + ':' + timetag[2:4]
            st = date + ' ' + tt_new
            starttime.append(np.datetime64(st))
        else:
            starttime.append(np.nan)
    for h in range(len(cal_dirs)):
        end_folder = cal_dirs[h].split("/")[-2]
        folder_tags = end_folder.split("_")[2:]
        timetag = None
        for k in range(len(folder_tags)):
            if len(folder_tags[k]) >= 4:
                if folder_tags[k].isnumeric():
                    timetag = folder_tags[k]
                elif folder_tags[k][:4].isnumeric():
                    timetag = folder_tags[k][:4]
        if timetag:
            if len(timetag) > 4:
                tt_new = timetag[:2] + ':' + timetag[2:4] + ':' + timetag[4:]
            else:
                tt_new = timetag[:2] + ':' + timetag[2:4]
            st = date + ' ' + tt_new
            starttime.append(np.datetime64(st))
        else: 
            starttime.append(np.nan)

    dc_dirs = data_dirs + cal_dirs
    fr = []
    for h in range(len(dc_dirs)):
        if obstype[h] == 'map':
            if zfps:
                fps = np.float(zfps)
            else:
                num_files = len(glob.glob(dc_dirs[h] + "*"))
                #### Time to make a wild-assed guess about the framerate.
                #### We'll say that if there's more than 15k frames in the data dir,
                #### Then it's probably 29.4 fps. Less is probably 64/60
                if num_files > 15000:
                    fps = 29.4
                else:
                    fps = 64./60.
        else:
            fps = 29.4
        fr.append(fps)
    for h in range(len(dc_dirs)):
        st = starttime[h]
        if not np.isnan(st):
            num_files = len(glob.glob(dc_dirs[h] + '*'))
            td = np.timedelta64(int(num_files/fr[h]),'s')
            et = st + td
            endtime.append(et)
        else:
            endtime.append(np.nan)
    zyla_obssum = np.rec.fromarrays([starttime,endtime,obstype],names=['start','end','type'])
    return zyla_obssum

def dummy_recarray():
    dummy_array = np.rec.fromarrays(
            [
                np.array([]),
                np.array([]),
                np.array([]),
            ],
            names=[
                'start',
                'end',
                'type'
            ]
    )
    return dummy_array

def automatic(indir, zfps, force_redo='n'):
    if force_redo == 'n':
        if any("observing_summary.jpg" in fl for fl in glob.glob(os.path.join(indir,"*"))):
            print("Obssum Plot Exists, exiting...")
            return

    params = {
        'image.origin': 'lower',
        'image.interpolation':'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 300,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'font.size': 12,
        'font.weight': 'bold',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
        'font.family': 'serif',
        'figure.facecolor': '#fef5e7'
    }

    plt.rcParams.update(params)

    dirlist = indir.split("/")
    obsdate = "-".join([x for x in dirlist if x.isnumeric()])
    ## We've got indir, odir, and obsdate. Make an observing plot.
    firs_dirs = glob.glob(
            indir + '**/level0/**/*firs*/',
            recursive=True)
    ibis_dirs = glob.glob(
            indir + "**/level0/**/*ibis*/",
            recursive=True)
    spinor_dirs = glob.glob(
            indir + '**/level0/**/*spinor*8542*/',
            recursive=True)
    gband_dir = glob.glob(
            indir + '**/level0/**/*gband*/',
            recursive=True)
    hsg_dirs = glob.glob(
            indir + '**/level0/**/*hsg*8542*/',
            recursive=True)
    zyla_dir = glob.glob(
            indir + '**/level0/**/*zyla*/',
            recursive=True)
    if len(zyla_dir) == 0:
        zyla_dir = glob.glob(
                indir + '**/level0/**/*Zyla*/',
                recursive=True)
    scanstart = []
    #### Putting together FIRS obssum
    for i in range(len(firs_dirs)):
        flist = glob.glob(firs_dirs[i] + '*0000')
        if (len(flist) > 0) and ('slitjaw' not in firs_dirs[i]):
            firs_dir = firs_dirs[i]
    if len(firs_dirs) == 0:
        firs_dir = []
    if len(firs_dir) > 0:
        firs_obssum = create_firs_obssum(firs_dir,obsdate)
        if not os.path.isdir(os.path.join(indir,"context_ims")):
            os.mkdir(os.path.join(indir,"context_ims"))
        use_slat = []
        use_slng = []
        use_stt = []
        use_ett = []
        use_scin = []
        use_stscin = []
        for i in range(len(firs_obssum.type)):
            if 'scan' in firs_obssum.type[i]:
                use_slat.append(firs_obssum.slat[i])
                use_slng.append(firs_obssum.slng[i])
                use_stt.append(firs_obssum.start[i].astype(str))
                use_ett.append(firs_obssum.end[i].astype(str))
                scanstart.append(firs_obssum.start[i].astype(str))
                use_scin.append(firs_obssum.scin[i].astype(str))
                use_stscin.append(firs_obssum.stdscin[i].astype(str))
        pointing_file = open(os.path.join(os.path.join(indir,'context_ims'), 'pointing_info.txt'), "w")
        pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
        for i in range(len(use_slat)):
           pointing_file.write(use_slat[i]+","+use_slng[i]+","+use_stt[i]+","+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+"\n")
        pointing_file.close()
    else:
        firs_obssum = dummy_recarray()
    #### IBIS obssum
    if len(ibis_dirs) > 0:
        ibis_obssum = create_ibis_obssum(ibis_dirs[0], obsdate)
        if (len(scanstart) == 0) & (len(ibis_obssum.start) > 0):
            if not os.path.isdir(os.path.join(indir, "context_ims")):
                os.mkdir(os.path.join(indir, "context_ims"))
            use_slat = []
            use_slng = []
            use_stt = []
            use_ett = []
            use_scin = []
            use_stscin = []
            for i in range(len(ibis_obssum.type)):
                if ibis_obssum.type[i] == 'scan':
                    use_slat.append(ibis_obssum.slat[i])
                    use_slng.append(ibis_obssum.slng[i])
                    use_stt.append(ibis_obssum.start[i].astype(str))
                    use_ett.append(ibis_obssum.end[i].astype(str))
                    scanstart.append(ibis_obssum.start[i].astype(str))
                    use_scin.append(ibis_obssum.scin[i].astype(str))
                    use_stscin.append(ibis_obssum.stdscin[i].astype(str))
            pointing_file = open(os.path.join(os.path.join(indir,"context_ims"), "pointing_info.txt"), "w")
            pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
            for i in range(len(use_slat)):
                pointing_file.write(use_slat[i] + "," + use_slng[i] + "," + use_stt[i] + "," + use_ett[i] + ','+use_scin[i]+','+use_stscin[i]+"\n")
            pointing_file.close()
    else:
        ibis_obssum = dummy_recarray()
    #### SPINOR obsusm
    if len(spinor_dirs) > 0:
        spinor_obssum = create_spinor_obssum(spinor_dirs[0],obsdate)
        if (len(scanstart) == 0) & (len(spinor_obssum.start) > 0):
            if not os.path.isdir(os.path.join(indir, "context_ims")):
                os.mkdir(os.path.join(indir, "context_ims"))
            
            use_slat = []
            use_slng = []
            use_stt = []
            use_ett = []
            use_scin = []
            use_stscin = []
            for i in range(len(spinor_obssum.type)):
                if spinor_obssum.type[i] == 'scan':
                    use_slat.append(spinor_obssum.slat[i])
                    use_slng.append(spinor_obssum.slng[i])
                    use_stt.append(spinor_obssum.start[i].astype(str))
                    use_ett.append(spinor_obssum.end[i].astype(str))
                    scanstart.append(spinor_obssum.start[i].astype(str))
                    use_scin.append(spinor_obssum.scin[i].astype(str))
                    use_stscin.append(spinor_obssum.stdscin[i].astype(str))
            pointing_file = open(os.path.join(os.path.join(indir,"context_ims"), "pointing_info.txt"), "w")
            pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
            for i in range(len(use_slat)):
                pointing_file.write(use_slat[i]+','+use_slng[i]+','+use_stt[i]+','+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+'\n')
            pointing_file.close()
        hsg_obssum = dummy_recarray()
    elif len(hsg_dirs) > 0:
        hsg_obssum = create_hsg_obssum(hsg_dirs[0],obsdate)
        if (len(scanstart) == 0) & (len(hsg_obssum.start) > 0):
            if not os.path.isdir(os.path.join(indir, "context_ims")):
                os.mkdir(os.path.join(indir, "context_ims"))
            use_slat = []
            use_slng = []
            use_stt = []
            use_ett = []
            use_scin = []
            use_stscin = []
            for i in range(len(hsg_obssum.type)):
                if hsg_obssum.type[i] == 'scan':
                    use_slat.append(hsg_obssum.slat[i])
                    use_slng.append(hsg_obssum.slng[i])
                    use_stt.append(hsg_obssum.start[i].astype(str))
                    use_ett.append(hsg_obssum.end[i].astype(str))
                    scanstart.append(hsg_obssum.start[i].astype(str))
                    use_scin.append(hsg_obssum.scin[i].astype(str))
                    use_stscin.append(hsg_obssum.stdscin[i].astype(str))
            if len(use_slat) > 10:
                use_slat = use_slat[:10]
                use_slng = use_slng[:10]
                use_stt = use_stt[:10]
            pointing_file = open(os.path.join(os.path.join(indir, "context_ims"), "pointing_info.txt"), "w")
            pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
            for i in range(len(use_slat)):
                pointing_file.write(use_slat[i]+','+use_slng[i]+','+use_stt[i]+','+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+'\n')
            pointing_file.close()        
        spinor_obssum = dummy_recarray()
    else:
        spinor_obssum = dummy_recarray()
        hsg_obssum = dummy_recarray()
    #### ROSA/Zyla obssum
    if len(gband_dir) > 0:
        rosa_obssum = create_rosa_obssum(gband_dir[0], obsdate)
        if len(zyla_dir) > 0:
            zyla_obssum = create_zyla_obssum(zyla_dir[0], obsdate, zfps)
            if len(zyla_obssum.type) >= 2:
                zyla_obssum = rosa_obssum
            else:
                zyla_obssum = dummy_recarray()

        else:
            zyla_obssum = dummy_recarray()
    else:
        rosa_obssum = dummy_recarray()
        if len(zyla_dir) > 0:
            zyla_obssum = create_zyla_obssum(zyla_dir[0],obsdate, zfps)
        else:
            zyla_obssum = dummy_recarray()
    #### Okay. Now we've got obssums for all of our *current* instruments.
    #### Eventually, I should come back and add IBIS for pre-Aug2019 data
    #### Eventually.
    #### Now we put it into a plot!
    fig = plt.figure(figsize = (10,8))
    gs = fig.add_gridspec(2,1,hspace=0.6)
    ax = fig.add_subplot(gs[0,0])
    ax_goes = fig.add_subplot(gs[1,0])

    all_starts = []
    all_ends = []
    barh_y_ctr = 0
    barh_ys = []
    labels = []
    #### FIRS FIRSt
    firs_types = ['sflt','dark','scan','pcal','targ','lgrd','pnhl','lflt']
    for i in range(len(firs_obssum.start)):
        if firs_types[0] in firs_obssum.type[i]:
            fc = 'C1'
        elif firs_types[-1] in firs_obssum.type[i]:
            fc = 'C1'
        elif firs_types[1] in firs_obssum.type[i]:
            fc = 'k'
        elif firs_types[2] in firs_obssum.type[i]:
            fc = 'C0'
        elif firs_types[3] in firs_obssum.type[i]:
            fc = 'C2'
        elif any(mystr in firs_obssum.type[i] for mystr in firs_types[4:7]):
            fc = 'C3'
        else:
            fc = 'C4'
        all_starts.append(firs_obssum.start[i])
        all_ends.append(firs_obssum.end[i])
        ax.broken_barh(
                [(firs_obssum.start[i],
                  firs_obssum.end[i]-firs_obssum.start[i])],
                  (barh_y_ctr,0.5),
                  facecolors=fc)
        if i == len(firs_obssum.start)-1:
            labels.append("FIRS")
            barh_ys.append(barh_y_ctr+0.25)
            barh_y_ctr += 0.75
    #### SPINOR SP....econd
    sp_types = ['sflat','scan','pcal','lflat']
    for i in range(len(spinor_obssum.start)):
        if sp_types[0] in spinor_obssum.type[i]:
            fc = 'C1'
        elif sp_types[-1] in spinor_obssum.type[i]:
            fc = 'C1'
        elif sp_types[1] in spinor_obssum.type[i]:
            fc = 'C0'
        elif sp_types[2] in spinor_obssum.type[i]:
            fc = 'C2'
        else:
            fc = 'C4'
        all_starts.append(spinor_obssum.start[i])
        all_ends.append(spinor_obssum.end[i])

        ax.broken_barh(
                [(spinor_obssum.start[i],
                  spinor_obssum.end[i]-spinor_obssum.start[i])],
                  (barh_y_ctr,0.5),
                  facecolors=fc)
        if i == len(spinor_obssum.start)-1:
            labels.append("SPINOR")
            barh_ys.append(barh_y_ctr+0.25)
            barh_y_ctr += 0.75

    #### HSG tHSGrd
    hsg_types = ['sflat','dark','scan','target','line','lflat']
    for i in range(len(hsg_obssum.start)):
        if hsg_types[0] in hsg_obssum.type[i]:
            fc = 'C1'
        elif hsg_types[-1] in hsg_obssum.type[i]:
            fc = 'C1'
        elif hsg_types[1] in hsg_obssum.type[i]:
            fc = 'k'
        elif hsg_types[2] in hsg_obssum.type[i]:
            fc = 'C0'
        elif any(mystr in hsg_obssum.type[i] for mystr in hsg_types[3:5]):
            fc = 'C3'
        else:
            fc = 'C4'
        all_starts.append(hsg_obssum.start[i])
        all_ends.append(hsg_obssum.end[i])

        ax.broken_barh(
                [(hsg_obssum.start[i],
                  hsg_obssum.end[i]-hsg_obssum.start[i])],
                  (barh_y_ctr,0.5),
                  facecolors=fc)
        if i == len(hsg_obssum.start)-1:
            labels.append("HSG")
            barh_ys.append(barh_y_ctr+0.25)
            barh_y_ctr += 0.75

    #### ROSA fouROth [these are getting out of hand. It's about to be a problem]
    rosa_types = ['flat','dark','map']
    for i in range(len(rosa_obssum.start)):
        if rosa_types[0] in rosa_obssum.type[i]:
            fc = 'C1'
        elif rosa_types[1] in rosa_obssum.type[i]:
            fc = 'k'
        elif rosa_types[2] in rosa_obssum.type[i]:
            fc = 'C0'
        else:
            fc = 'C4'
        all_starts.append(rosa_obssum.start[i])
        all_ends.append(rosa_obssum.end[i])

        ax.broken_barh(
                [(rosa_obssum.start[i],
                  rosa_obssum.end[i] - rosa_obssum.start[i])],
                  (barh_y_ctr,0.5),
                  facecolors=fc)
        if i == len(rosa_obssum.start)-1:
            labels.append("ROSA")
            barh_ys.append(barh_y_ctr+0.25)
            barh_y_ctr += 0.75

    #### zyla zyla fukkit.
    zyla_types = ['flat','dark','map','dot','pin','line','target']
    for i in range(len(zyla_obssum.start)):
        if not np.isnan(zyla_obssum.start[i]):
            if zyla_types[0] in zyla_obssum.type[i]:
                fc = 'C1'
            elif zyla_types[1] in zyla_obssum.type[i]:
                fc = 'k'
            elif zyla_types[2] in zyla_obssum.type[i]:
                fc = 'C0'
            elif any(mystr in zyla_obssum.type[i] for mystr in zyla_types[3:]):
                fc = 'C3'
            else:
                fc = 'C4'
            all_starts.append(zyla_obssum.start[i])
            all_ends.append(zyla_obssum.end[i])
            dur = zyla_obssum.end[i] - zyla_obssum.start[i]
            if dur < np.timedelta64(10,'s'):
                dur = np.timedelta64(10,'s')
            ax.broken_barh(
                    [(zyla_obssum.start[i],
                      dur)],
                      (barh_y_ctr,0.5),
                      facecolors=fc)
            if i == len(zyla_obssum.start)-1:
                labels.append("ZYLA")
                barh_ys.append(barh_y_ctr+0.25)
                barh_y_ctr += 0.75

    #### IBIS after all
    ibis_types = ['sflat','dark','scan','targ','lgrd','othr']
    for i in range(len(ibis_obssum.start)):
        if ibis_types[0] in ibis_obssum.type[i]:
            fc = 'C1'
        elif ibis_types[1] in ibis_obssum.type[i]:
            fc = 'k'
        elif ibis_types[2] in ibis_obssum.type[i]:
            fc = 'C0'
        elif ibis_types[3] in ibis_obssum.type[i]:
            fc = 'C3'
        elif ibis_types[4] in ibis_obssum.type[i]:
            fc = 'C3'
        else:
            fc = 'C4'
        all_starts.append(ibis_obssum.start[i])
        all_ends.append(ibis_obssum.end[i])
        ax.broken_barh(
                [(ibis_obssum.start[i],
                  ibis_obssum.end[i] - ibis_obssum.start[i])],
                (barh_y_ctr, 0.5),
                facecolors=fc)
        if i == len(ibis_obssum.start)-1:
            labels.append("IBIS")
            barh_ys.append(barh_y_ctr+0.25)


    ax.set_title('DST Observations for '+obsdate,weight='bold')
    ax.set_yticks(barh_ys,labels=labels)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    #### Dummy objects for legend
    if len(firs_obssum.start) > 0:
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='C1',label='Flat')
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='k',label='Dark')
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='C0',label='Scan/Map')
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='C2',label='Polcal')
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='C3',label='Target/Linegrid')
        ax.broken_barh([(firs_obssum.start[0],0)],(0,0),facecolors='C4',label='Uncatagorized')
        ax.legend(loc='center',bbox_to_anchor=(0.5,-0.25),ncols=3)
    ####
    all_starts = np.array(all_starts)
    all_ends = np.array(all_ends)
    ax.set_xlim(all_starts.min(),all_ends.max())
    trange = a.Time(all_starts.min().astype(str),all_ends.max().astype(str))
    search = Fido.search(trange,a.Instrument('XRS'))
    if search.file_num == 0:
        goes_timestamps = []
        short = []
        long = []
    else:
        search = search[0,0]
        goes_dl = Fido.fetch(search,path=indir)
        goes_timeseries = ts.TimeSeries(goes_dl)
        goes_data = goes_timeseries.to_dataframe()
        goes_data = goes_data[(goes_data['xrsa_quality']==0)&(goes_data['xrsb_quality']==0)]
        goes_timeseries = ts.TimeSeries(goes_data,goes_timeseries.meta,goes_timeseries.units)
        truncated = goes_timeseries.truncate(all_starts.min().astype(str),all_ends.max().astype(str))
        goes_timestamps = truncated.time.value.astype('datetime64[s]')
        short = np.log10(truncated.quantity('xrsa').value)
        long = np.log10(truncated.quantity('xrsb').value)
    ax_goes.plot(goes_timestamps,short,label='0.5--4 AA')
    ax_goes.plot(goes_timestamps,long,label='1--8 AA')
    ax_goes.set_ylim(-9,-2)
    ax_goes.set_yticks([-9,-8,-7,-6,-5,-4,-3,-2])
    ax_goes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_goes.grid(axis='y',linestyle='--',linewidth=2)
    ax_goes.legend(loc='center',bbox_to_anchor=(0.5,-0.25),ncols=2)
    ax_goes.set_ylabel("log(Flux [W/m2])",weight='bold')
    ax_goes.set_xlim(all_starts.min(),all_ends.max())
    if len(goes_timestamps) == 0:
        ax_goes.text(
                np.datetime64(all_starts.min()) + (np.datetime64(all_ends.max()) - np.datetime64(all_starts.min()))/2,
                -5.5,
                "NO GOES DATA FOUND",
                weight="bold",
                ha='center',va='center')
    sec_ax=ax_goes.secondary_yaxis('right')
    sec_ax.set_ylabel("Flare Class",rotation=270,labelpad=25,weight='bold')
    sec_ax.set_yticks([-7.5,-6.5,-5.5,-4.5,-3.5],
                   labels=['A','B','C','M','X'])
    ax_goes.set_title("Simultaneous GOES Data",weight='bold')
    fig.savefig(os.path.join(indir,'observing_summary.jpg'),bbox_inches='tight')
