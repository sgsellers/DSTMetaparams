"""
Series of functions for determining the telescope:
    1.) Instruments Used
    2.) Wavelengths Observed
    3.) Timing of observations
    4.) Solar Coordinates during observing
These can be passed to targets_observed.py to determine targets, events, and co-observations
"""

import numpy as np
import glob
import astropy.io.fits as fits
import os
import warnings


def create_firs_obssum(baseDir, obsDate):
    """Finds FIRS level-0 Spectral files in baseDir, and sorts them by type.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD
    obsDate : str
        Date observations were taken. Required, as sometimes polcals from alternate dates are used.

    Returns:
    --------
    firsObssum : numpy.rec.recarray
        Records array of FIRS obssum parameters
    """

    obsDate = np.datetime64(obsDate)

    firsSearch = os.path.join(
        baseDir,
        'level0',
        '**',
        "firs.2.*"
    )
    firsList = sorted(glob.glob(firsSearch, recursive=True))

    # Get unique observing series
    firsObs = []
    for file in firsList:
        firsObs.append('.'.join(file.split(".")[:-2]))
    firsObs = list(set(firsObs))

    starttime = []
    endtime = []
    obstype = []
    slat = []
    slon = []
    meanSee = []
    stdSee = []
    for obs in firsObs:
        fileSeries = sorted(glob.glob(obs + '*'))
        firstFile = fits.open(fileSeries[0])
        lastFile = fits.open(fileSeries[-1])
        """
        FIRS obstypes are stored as "COMMENTS" in individual fits file headers.
        Check that the COMMENT header keyword exits, and that it isn't empty
        """
        if "COMMENT" not in firstFile[0].header.keys():
            continue
        elif firstFile[0].header['COMMENT'] == '':
            continue
        """
        Additionally, we take this moment to weed out polcals from other days
        """
        if any(s in firstFile[0].header['COMMENT'][0].lower() for s in ("pcal", "polcal")):
            obsStart = np.datetime64(firstFile[0].header['OBS_STAR'])
            if (obsStart > (obsDate + np.timedelta64(1, "D"))) or (obsStart < obsDate):
                continue

        obstype.append(firstFile[0].header['COMMENT'][0].lower().split(" ")[0])
        starttime.append(np.datetime64(firstFile[0].header['OBS_STAR']))
        endtime.append(np.datetime64(lastFile[0].header['OBS_END']))
        if "DST_SLAT" in list(firstFile[0].header.keys()):
            slat.append(float(firstFile[0].header['DST_SLAT']))
            slon.append(float(firstFile[0].header['DST_SLNG']))
        else:
            slat.append(0)
            slon.append(0)
        see = []
        for file in fileSeries:
            f = fits.open(file)
            if "DST_SEE" in list(f[0].header.keys()):
                see.append(
                    f[0].header['DST_SEE']
                )
            f.close()
        meanSee.append(np.nanmean(see))
        stdSee.append(np.nanstd(see))
        firstFile.close()
        lastFile.close()
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slon = np.array(slon)
    meanSee = np.array(meanSee)
    stdSee = np.array(stdSee)
    firsObssum = np.rec.fromarrays(
        [
            starttime,
            endtime,
            obstype,
            slat,
            slon,
            meanSee,
            stdSee
        ],
        names=[
            'start',
            'end',
            'type',
            'slat',
            'slon',
            'scin',
            'stdscin'
        ]
    )
    return firsObssum


def create_ibis_obssum(baseDir):
    """Finds IBIS level-0 Spectral files in baseDir, and sorts them by type.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD

    Returns:
    --------
    ibisObssum : numpy.rec.recarray
        Records array of IBIS obssum parameters
    centralWavelength : list
        List of unique wavelength observed
    """

    ibisSearch = os.path.join(
        baseDir,
        'level0',
        '**',
        "*ibis*",
        '*'
    )
    ibisSubdirs = sorted(glob.glob(ibisSearch + '/', recursive=True))

    starttime = []
    endtime = []
    obstype = []
    slat = []
    slon = []
    meanSee = []
    stdSee = []
    centralWavelength = []
    for directory in ibisSubdirs:
        dataDir = glob.glob(
            os.path.join(directory, '*') + '/'
        )
        for folder in dataDir:
            files = sorted(
                glob.glob(
                    os.path.join(folder, '*.fits*')
                )
            )
            try:
                firstFile = fits.open(files[0])
            except (Exception,):
                try:
                    firstFile = fits.open(files[1])
                    warnings.warn("Corrupted File: {0}".format(files[0]))
                except (Exception,):
                    continue
            if len(files) > 1:
                try:
                    lastFile = fits.open(files[-1])
                except (Exception,):
                    lastFile = fits.open(files[-2])
                    warnings.warn("Corrupted File: {0}".format(files[-1]))
            else:
                lastFile = firstFile
            starttime.append(np.datetime64(firstFile[1].header['DATE-OBS']))
            endtime.append(np.datetime64(lastFile[-1].header['DATE-END']))
            try:
                slat.append(float(firstFile[1].header['DST_SLAT']))
                slon.append(float(firstFile[1].header['DST_SLNG']))
            except (Exception,):
                # Apparently, this can break sometimes.
                slat.append(0)
                slon.append(0)
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
            firstFile.close()
            lastFile.close()
            see = []
            wvls = []
            for file in files:
                try:
                    workingFile = fits.open(file)
                    for j in range(1, len(workingFile)):
                        try:
                            scmon = float(workingFile[j].header['DST_SEE'])
                            wvls.append(
                                str(
                                    round(
                                        workingFile[j].header['WAVELNTH'] - workingFile[j].header['REL_WAVE']
                                    )
                                )
                            )
                        except (Exception,):
                            scmon = np.nan
                            warnings.warn(
                                "No Wavelength/Scintillation Values for Extension {0} in File: {1}".format(
                                    j, file
                                )
                            )
                        see.append(scmon)
                    workingFile.close()
                except (Exception,):
                    warnings.warn("Corrupted File: {0}".format(file))
                    pass
            meanSee.append(np.nanmean(see))
            stdSee.append(np.nanmean(see))
            for filt in wvls:
                centralWavelength.append(filt)

    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slon = np.array(slon)
    meanSee = np.array(meanSee)
    stdSee = np.array(stdSee)
    ibisObssum = np.rec.fromarrays(
        [
            starttime,
            endtime,
            obstype,
            slat,
            slon,
            meanSee,
            stdSee
        ],
        names=[
            'start',
            'end',
            'type',
            'slat',
            'slon',
            'scin',
            'stdscin'
        ]
    )
    centralWavelength = list(set(centralWavelength))
    return ibisObssum, centralWavelength


def create_spinor_obssum(baseDir, obsDate):
    """Finds SPINOR Level-0 Spectral files in baseDir, and sorts them by type.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD
    obsDate : str
        Date observations were taken. Required, as sometimes polcals from alternate dates are used.

    Returns:
    --------
    spinorObssum : numpy.rec.recarray
        Records array of SPINOR obssum parameters
    centralWavelengths : list
        List of spectral windows observed by SPINOR
    """

    obsDate = np.datetime64(obsDate)

    spinorDirs = sorted(
        glob.glob(
            os.path.join(
                baseDir,
                'level0',
                '**',
                "*spinor*/"
            ), recursive=True
        )
    )
    spinorDirs = [directory for directory in spinorDirs if "slitjaw" not in directory]
    if len(spinorDirs) == 0:
        spinorObssum = np.rec.fromarrays(
            [
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0)
            ],
            names=[
                'start',
                'end',
                'type',
                'slat',
                'slon',
                'scin',
                'stdscin'
            ]
        )
        return spinorObssum, []
    centralWavelengths = []
    for directory in spinorDirs:
        dirName = directory.split("/")[-2]
        dirTags = dirName.split("_")
        for tag in dirTags:
            if (len(tag) == 4) & (tag.isnumeric()):
                centralWavelengths.append(tag)
    spinorFiles = sorted(
        glob.glob(
            os.path.join(
                spinorDirs[0],
                '*.fits'
            )
        )
    )

    starttime = []
    endtime = []
    obstype = []
    slat = []
    slon = []
    meanSee = []
    stdSee = []
    for file in spinorFiles:
        try:
            spinorFits = fits.open(file)
        except (Exception,):
            warnings.warn("Corrupted File: {0}".format(file))
            continue
        obsStart = np.datetime64(spinorFits[1].header['DATE-OBS'])
        try:
            obsEnd = np.datetime64(spinorFits[-1].header['DATE-OBS'])
        except (Exception,):
            warnings.warn("Corrupted Extension {0} in file: {1}".format('-1', file))
            try:
                obsEnd = np.datetime64(spinorFits[-2].header['DATE-OBS'])
            except (Exception,):
                obsEnd = obsStart
                warnings.warn("Spinor is a real piece of shit.")
        if (obsStart > (obsDate + np.timedelta64(1, "D"))) or (obsStart < obsDate):
            continue
        starttime.append(obsStart)
        endtime.append(obsEnd)
        slat.append(spinorFits[1].header['DST_SLAT'])
        slon.append(spinorFits[1].header['DST_SLNG'])
        see = []
        for j in range(1, len(spinorFits)):
            try:
                scmon = spinorFits[j].header['DST_SEE']
            except (Exception,):
                warnings.warn("No scintillation in extension {0} of file: {1}".format(j, file))
                scmon = np.nan
            see.append(scmon)
        meanSee.append(np.nanmean(see))
        stdSee.append(np.nanstd(see))
        if '.flat.' in file:
            obstype.append('sflat')
        elif '.lamp.flat.' in file:
            obstype.append('lflat')
        elif '.cal.' in file:
            obstype.append('pcal')
        else:
            if 'PT4_FS' not in list(spinorFits[1].header.keys()):
                obstype.append("othr")
                spinorFits.close()
                continue
            if 'USER1' in spinorFits[1].header['PT4_FS']:
                obstype.append('scan')
            else:
                obstype.append(spinorFits[1].header['PT4_FS'].lower())
        spinorFits.close()
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slon = np.array(slon)
    meanSee = np.array(meanSee)
    stdSee = np.array(stdSee)
    spinorObssum = np.rec.fromarrays(
        [
            starttime,
            endtime,
            obstype,
            slat,
            slon,
            meanSee,
            stdSee
        ],
        names=[
            'start',
            'end',
            'type',
            'slat',
            'slon',
            'scin',
            'stdscin'
        ]
    )
    return spinorObssum, centralWavelengths


def create_hsg_obssum(baseDir, obsDate):
    """Finds HSG Level-0 Spectral files in baseDir, and sorts them by type.
    Somewhat complicated by the fact that, for certain observing series, HSG was run via
    the SPINOR GUI, rather than the ICC. This changes the file structure and naming conventions.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD
    obsDate : str
        Date observations were taken. Required, as sometimes polcals from alternate dates are used.

    Returns:
    --------
    hsgObssum : numpy.rec.recarray
        Records array of HSG obssum parameters
    centralWavelengths : list
        List of spectral windows observed by HSG
    """

    obsDate = np.datetime64(obsDate)

    hsgDirs = sorted(
        glob.glob(
            os.path.join(
                baseDir,
                'level0',
                '**',
                "*hsg*/"
            ), recursive=True
        )
    )
    hsgDirs = [directory for directory in hsgDirs if "slitjaw" not in directory]
    if len(hsgDirs) == 0:
        hsgObssum = np.rec.fromarrays(
            [
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0)
            ],
            names=[
                'start',
                'end',
                'type',
                'slat',
                'slon',
                'scin',
                'stdscin'
            ]
        )
        return hsgObssum, []
    centralWavelengths = []
    for directory in hsgDirs:
        dirName = directory.split("/")[-2]
        dirTags = dirName.split("_")
        for tag in dirTags:
            if (len(tag) == 4) & (tag.isnumeric()):
                centralWavelengths.append(tag)
    hsgFiles = sorted(
        glob.glob(
            os.path.join(
                hsgDirs[0],
                '*.fits'
            )
        )
    )
    starttime = []
    endtime = []
    obstype = []
    slat = []
    slon = []
    meanSee = []
    stdSee = []

    for file in hsgFiles:
        hsgFits = fits.open(file)
        obsStart = np.datetime64(hsgFits[0].header['DATE-BGN'])
        if (obsStart > (obsDate + np.timedelta64(1, "D"))) or (obsStart < obsDate):
            continue
        try:
            obsEnd = np.datetime64(hsgFits[-1].header['DATE-OBS'])
        except (Exception,):
            warnings.warn("Corrupted Extension {0} in file: {1}".format('-1', file))
            try:
                obsEnd = np.datetime64(hsgFits[-2].header['DATE-OBS'])
            except (Exception,):
                obsEnd = obsStart
                warnings.warn("HSG, when run like Spinor is a real piece of shit. Also, other times.")
        starttime.append(obsStart)
        endtime.append(obsEnd)
        slat.append(hsgFits[1].header['DST_SLAT'])
        slon.append(hsgFits[1].header['DST_SLNG'])
        if any(substr in file for substr in ('sun.flat', 'solar_flat')):
            obstype.append('sflat')
        elif any(substr in file for substr in ('lamp_flat', 'lamp.flat')):
            obstype.append('lflat')
        elif 'dark' in file:
            obstype.append('dark')
        elif 'linegrid' in file:
            obstype.append('lgrd')
        elif 'target' in file:
            obstype.append('targ')
        elif 'cal' in file:
            obstype.append('pcal')
        elif 'scan' in file:
            obstype.append('scan')
        elif 'map' in file:
            if 'USER1' in hsgFits[1].header['PT4_FS']:
                obstype.append('scan')
            else:
                obstype.append(hsgFits[1].header['PT4_FS'].lower())
        else:
            obstype.append('othr')
        see = []
        for j in range(1, len(hsgFits)):
            try:
                scmon = hsgFits[j].header['DST_SEE']
            except (Exception,):
                scmon = np.nan
            see.append(scmon)
        meanSee.append(np.nanmean(see))
        stdSee.append(np.nanstd(see))
        hsgFits.close()
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)
    slat = np.array(slat)
    slon = np.array(slon)
    meanSee = np.array(meanSee)
    stdSee = np.array(stdSee)
    hsgObssum = np.rec.fromarrays(
        [
            starttime,
            endtime,
            obstype,
            slat,
            slon,
            meanSee,
            stdSee
        ],
        names=[
            'start',
            'end',
            'type',
            'slat',
            'slon',
            'scin',
            'stdscin'
        ]
    )
    return hsgObssum, centralWavelengths


def create_rosa_obssum(baseDir, obsDate):
    """Finds ROSA Level-0 files in baseDir, and sorts them by type.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD
    obsDate : str
        Date observations were taken. Required, as sometimes polcals from alternate dates are used.

    Returns:
    --------
    rosaObssum : numpy.rec.recarray
        Records array of ROSA obssum parameters
    filtersUsed : list
        List of filters observed by ROSA
    """

    allowedFilters = ['gband', '4170', 'cak', '3500']
    filesPerFilter = []
    filtersUsed = []
    for wave in allowedFilters:
        rosaSearch = os.path.join(
            baseDir,
            'level0',
            '**',
            '*'+wave+'*',
            '*.fit*'
        )
        filterFiles = sorted(glob.glob(rosaSearch, recursive=True))
        filesPerFilter.append(len(filterFiles))
        if len(filterFiles) > 0:
            filtersUsed.append(wave)
    bestFilter = allowedFilters[filesPerFilter.index(np.nanmax(filesPerFilter))]
    # Okay. This is getting fucked. Learn regexes already, christ almighty.
    rosaFiles = sorted(
        glob.glob(
            os.path.join(
                baseDir,
                'level0',
                '**',
                '*'+bestFilter+'*',
                '*.fit*'
            ), recursive=True
        )
    )
    rosaObs = []
    for file in rosaFiles:
        tags = file.split('.')
        if len(tags) == 5:
            # Get rid of the .fz tag for fpacked files
            tags = tags[:-2]
        else:
            tags = tags[:-1]
        obs = '.'.join(tags)
        obs = obs[:-5]
        rosaObs.append(obs)
    rosaObs = list(set(rosaObs))
    starttime = []
    endtime = []
    obstype = []
    for obs in rosaObs:
        seriesFiles = sorted(glob.glob(obs + '*'))
        firstFile = fits.open(seriesFiles[0])
        lastFile = fits.open(seriesFiles[-1])
        try:
            obsStart = np.datetime64(firstFile[1].header['DATE'])
        except (Exception,):
            warnings.warn("File no good, " + seriesFiles[0])
            try:
                firstFile = fits.open(seriesFiles[1])
                obsStart = np.datetime64(firstFile[1].header['DATE'])
            except (Exception,):
                continue
        if (obsStart > (np.datetime64(obsDate) + np.timedelta64(1, "D"))) or (obsStart < np.datetime64(obsDate)):
            continue
        # ROSA's exit from observing isn't always graceful.
        # Rather than take the last extension of the last file,
        # We go by the number of extensions and the framerate.
        dframe = int(lastFile[0].header['HIERARCH FRAME RATE'].split(" ")[0])
        td = np.timedelta64(int((len(lastFile) - 1) * dframe), 'ms')
        starttime.append(obsStart)
        endtime.append(np.datetime64(lastFile[1].header['DATE']) + td)
        firstFile.close()
        lastFile.close()
        if 'flat' in obs:
            obstype.append('sflat')
        elif 'dark' in obs:
            obstype.append('dark')
        else:
            # No way of separating out targets/linegrids/etc without considerable extra
            # effort. Instead, we fudge it -- if the obsduration is less than 5 minutes,
            # we call it an 'other' type. Otherwise, it's 'map' type.
            if (endtime[-1] - obsStart) < np.timedelta64(5, "m"):
                obstype.append('othr')
            else:
                obstype.append('map')
    starttime = np.array(starttime)
    endtime = np.array(endtime)
    obstype = np.array(obstype)

    rosaObssum = np.rec.fromarrays(
        [
            starttime,
            endtime,
            obstype
        ],
        names=[
            'start',
            'end',
            'type'
        ]
    )
    return rosaObssum, filtersUsed


def create_zyla_obssum(baseDir, obsDate):
    """
    If I never have to write this fucking function again, that would be great.
    Once again, Zyla has no way of recording:
        1.) Framerate
        2.) Datatype
        3.) Observing Time
        4.) Pointing information
        5.) The data shape
    We must guess at all of these.
    Compared to previous versions of this function, I've removed the zfps kwarg.
    Instead, we will assume a constant 29.4fps for calibration datasets.
    For science obs, we start with 64 images/60 seconds, but if this causes an overlap with the next
    observing series, we will run down the following possible framerates:
        1.) 29.4
        2.) 64-image burst every 60s
        3.) 64-image burst every 30s
        4.) 64-image burst every 16s
        5.) 64-image burst every 8s
    We will cease iteration when we find a framerate that causes no overlaps.

    Parameters:
    -----------
    baseDir : str
        Path to observing day.
    obsDate : str
        Date of observations.

    Returns:
    --------
    zylaObssum : np.rec.recarray
        Record array of Zyla observing summary params.
    """

    framerates = [64./60, 64./30, 64./8, 29.4, 'skip']

    obsDir = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*obs*/'), recursive=True)
    obsDir2 = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*Obs*/'), recursive=True)
    filDir = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*fil*/'), recursive=True)
    limbDir = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*limb*/'), recursive=True)
    arDir = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*_ar*/'), recursive=True)
    arDir2 = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*_AR*/'), recursive=True)
    burDir = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*burst*/'), recursive=True)
    dataDirs = list(set(obsDir + obsDir2 + filDir + limbDir + arDir + arDir2 + burDir))

    calTags = ['flat', 'dark', 'dot', 'pin', 'error', 'line', 'target', 'test']
    calType = ['flat', 'dark', 'dot', ' pin', 'line', 'target', 'test']

    allDirs = glob.glob(os.path.join(
        baseDir, 'level0', '**/DBJ*/'), recursive=True)
    calDirs = []
    obstype = []
    altDataDirs = []
    for subdir in allDirs:
        if not any(x in subdir for x in calTags):
            altDataDirs.append(subdir)
        else:
            for tag in calType:
                if tag in subdir:
                    obstype.append(tag)
                    calDirs.append(subdir)
    if len(dataDirs) == 0:
        dataDirs = altDataDirs

    useDirs = dataDirs + calDirs
    dataOtype = ['map' for subd in dataDirs]
    obstype = dataOtype + obstype

    starttime = []
    endtime = []
    nfiles = []
    # Find approximate starttimes
    for subdir in useDirs:
        nfiles.append(len(glob.glob(os.path.join(subdir, "*.dat*"))))
        workingFolder = subdir.split("/")[-2]
        folderTags = workingFolder.split("_")[2:]
        timetag = None
        for tag in folderTags:
            if len(tag) >= 4:
                if tag.isnumeric():
                    timetag = tag
                elif tag[:4].isnumeric():
                    timetag = tag[:4]
        if timetag:
            if len(timetag) > 4:
                useTime = timetag[:2] + ":" + timetag[2:4] + ":" + timetag[4:]
            else:
                useTime = timetag[:2] + ":" + timetag[2:4]
            starttime.append(np.datetime64(obsDate + " " + useTime))
        else:
            starttime.append(np.nan)
    starttime = np.array(starttime)
    obstype = np.array(obstype)
    nfiles = np.array(nfiles)
    useDirs = np.array(useDirs)

    startSort = np.argsort(starttime)
    starttime = starttime[startSort]
    obstype = obstype[startSort]
    nfiles = nfiles[startSort]
    useDirs = useDirs[startSort]

    for i in range(len(starttime)):
        if not np.isnan(starttime[i]):
            if i == (len(starttime) - 1):
                overflowTime = np.datetime64(obsDate + " 23:00")
            else:
                overflowTime = starttime[i+1]
            if 'map' in obstype[i]:
                # Kludge just in case it's clearly a 29.4fps dataset
                # 20k frames is only 11 minutes at 29.4 fps, but it's
                # Five hours at the 64frames/60s cadence and
                # Two and a half hours at 64frames/30s.
                # So if it's got a bunch of frames, it's probably high cadence.
                # Everybody else has to guess.
                if nfiles[i] >= 20000:
                    framerateGuess = 3
                else:
                    framerateGuess = 0
                projectedEnd = starttime[i] + np.timedelta64(
                    int(
                        nfiles[i]/framerates[framerateGuess]
                    ), 's'
                )
                while projectedEnd > overflowTime:
                    if framerates[framerateGuess] == 'skip':
                        projectedEnd = overflowTime
                        break
                    else:
                        projectedEnd = starttime[i] + np.timedelta64(
                            int(
                                nfiles[i]/framerates[framerateGuess]
                            ), 's'
                        )
                    framerateGuess += 1
            # Cals are usually run at 29.4 FPS, and also the precise timing doesn't matter
            else:
                projectedEnd = starttime[i] + np.timedelta64(
                    int(
                        nfiles[i] / 29.4
                    ), 's'
                )
        else:
            projectedEnd = np.nan
        endtime.append(projectedEnd)
    endtime = np.array(endtime)
    useDirs = np.array(useDirs)
    zylaObssum = np.rec.fromarrays(
        [starttime, endtime, obstype, useDirs],
        names=['start', 'end', 'type', 'directory']
    )
    return zylaObssum
