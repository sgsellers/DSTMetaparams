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
    obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])

    return obssum

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
    obssum = np.rec.fromarrays([starttime, endtime, obstype, slat, slng,mean_see,std_see], names=['start','end','type','slat','slng','scin','stdscin'])
    return obssum
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
    obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])
    return obssum

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
    obssum = np.rec.fromarrays([starttime,endtime,obstype,slat,slng,mean_see,std_see],names=['start','end','type','slat','slng','scin','stdscin'])
    return obssum

def automatic(indir):
    obssum = None
    dirlist = indir.split("/")
    obsdate = "-".join([x for x in dirlist if x.isnumeric()])
    ## We've got indir, odir, and obsdate. Make an observing plot.
    scanstart = []
    #### Putting together FIRS obssum
    firs_dirs = glob.glob(
            indir + '**/level0/**/*firs*/',
            recursive=True)
    for i in range(len(firs_dirs)):
        flist = glob.glob(firs_dirs[i] + '*0000')
        if (len(flist) > 0) and ('slitjaw' not in firs_dirs[i]):
            firs_dir = firs_dirs[i]
    if len(firs_dirs) == 0:
        firs_dir = []
    if len(firs_dir) > 0:
        obssum = create_firs_obssum(firs_dir,obsdate)
        if not os.path.isdir(os.path.join(indir,"context_ims")):
            os.mkdir(os.path.join(indir,"context_ims"))
        use_slat = []
        use_slng = []
        use_stt = []
        use_ett = []
        use_scin = []
        use_stscin = []
        for i in range(len(obssum.type)):
            if 'scan' in obssum.type[i]:
                use_slat.append(obssum.slat[i])
                use_slng.append(obssum.slng[i])
                use_stt.append(obssum.start[i].astype(str))
                use_ett.append(obssum.end[i].astype(str))
                scanstart.append(obssum.start[i].astype(str))
                use_scin.append(obssum.scin[i].astype(str))
                use_stscin.append(obssum.stdscin[i].astype(str))
        pointing_file = open(os.path.join(os.path.join(indir,'context_ims'), 'pointing_info.txt'), "w")
        pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
        for i in range(len(use_slat)):
           pointing_file.write(use_slat[i]+","+use_slng[i]+","+use_stt[i]+","+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+"\n")
        pointing_file.close()

    if obssum is None:
        ibis_dirs = glob.glob(
                indir + "**/level0/**/*ibis*/",
                recursive=True)
        #### IBIS obssum
        if len(ibis_dirs) > 0:
            obssum = create_ibis_obssum(ibis_dirs[0], obsdate)
            if (len(scanstart) == 0) & (len(obssum.start) > 0):
                if not os.path.isdir(os.path.join(indir, "context_ims")):
                    os.mkdir(os.path.join(indir, "context_ims"))
                use_slat = []
                use_slng = []
                use_stt = []
                use_ett = []
                use_scin = []
                use_stscin = []
                for i in range(len(obssum.type)):
                    if obssum.type[i] == 'scan':
                        use_slat.append(obssum.slat[i])
                        use_slng.append(obssum.slng[i])
                        use_stt.append(obssum.start[i].astype(str))
                        use_ett.append(obssum.end[i].astype(str))
                        scanstart.append(obssum.start[i].astype(str))
                        use_scin.append(obssum.scin[i].astype(str))
                        use_stscin.append(obssum.stdscin[i].astype(str))
                pointing_file = open(os.path.join(os.path.join(indir,"context_ims"), "pointing_info.txt"), "w")
                pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
                for i in range(len(use_slat)):
                    pointing_file.write(use_slat[i] + "," + use_slng[i] + "," + use_stt[i] + "," + use_ett[i] + ','+use_scin[i]+','+use_stscin[i]+"\n")
                pointing_file.close()
    if obssum is None:
        spinor_dirs = glob.glob(
                indir + '**/level0/**/*spinor_sarnoff_8542*/',
                recursive=True)

        #### SPINOR obsusm
        if len(spinor_dirs) > 0:
            obssum = create_spinor_obssum(spinor_dirs[0],obsdate)
            if (len(scanstart) == 0) & (len(obssum.start) > 0):
                if not os.path.isdir(os.path.join(indir, "context_ims")):
                    os.mkdir(os.path.join(indir, "context_ims"))
                use_slat = []
                use_slng = []
                use_stt = []
                use_ett = []
                use_scin = []
                use_stscin = []
                for i in range(len(obssum.type)):
                    if obssum.type[i] == 'scan':
                        use_slat.append(obssum.slat[i])
                        use_slng.append(obssum.slng[i])
                        use_stt.append(obssum.start[i].astype(str))
                        use_ett.append(obssum.end[i].astype(str))
                        scanstart.append(obssum.start[i].astype(str))
                        use_scin.append(obssum.scin[i].astype(str))
                        use_stscin.append(obssum.stdscin[i].astype(str))
                pointing_file = open(os.path.join(os.path.join(indir,"context_ims"), "pointing_info.txt"), "w")
                pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
                for i in range(len(use_slat)):
                    pointing_file.write(use_slat[i]+','+use_slng[i]+','+use_stt[i]+','+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+'\n')
                pointing_file.close()

    if obssum is None:
        hsg_dirs = glob.glob(
                indir + '**/level0/**/*hsg*8542*/',
                recursive=True)
        if len(hsg_dirs) > 0:
            obssum = create_hsg_obssum(hsg_dirs[0],obsdate)
            if (len(scanstart) == 0) & (len(obssum.start) > 0):
                if not os.path.isdir(os.path.join(indir, "context_ims")):
                    os.mkdir(os.path.join(indir, "context_ims"))
                use_slat = []
                use_slng = []
                use_stt = []
                use_ett = []
                use_scin = []
                use_stscin = []
                for i in range(len(obssum.type)):
                    if obssum.type[i] == 'scan':
                        use_slat.append(obssum.slat[i])
                        use_slng.append(obssum.slng[i])
                        use_stt.append(obssum.start[i].astype(str))
                        use_ett.append(obssum.end[i].astype(str))
                        scanstart.append(obssum.start[i].astype(str))
                        use_scin.append(obssum.scin[i].astype(str))
                        use_stscin.append(obssum.stdscin[i].astype(str))
                if len(use_slat) > 10:
                    use_slat = use_slat[:10]
                    use_slng = use_slng[:10]
                    use_stt = use_stt[:10]
                pointing_file = open(os.path.join(os.path.join(indir, "context_ims"), "pointing_info.txt"), "w")
                pointing_file.write("SLAT,SLON,TIME,END,MSCIN,SSCIN\n")
                for i in range(len(use_slat)):
                    pointing_file.write(use_slat[i]+','+use_slng[i]+','+use_stt[i]+','+use_ett[i]+','+use_scin[i]+','+use_stscin[i]+'\n')
                pointing_file.close()
