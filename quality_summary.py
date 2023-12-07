import numpy as np
from . import observation_summary as obssum
from . import seeing_quality as seeing


def _find_nearest(array, value):
    """Determines the index of the closest value in an array to a specified other value

    Parameters
    ----------
    array : array-like
        An array of int/float values
    value : int,float
        A value that we will check for in the array

    Returns
    -------
    idx : int
        The index of the input array where the closest value is found
    """
    idx = (np.abs(array-value)).argmin()
    return idx


def _correct_daylight_savings(timestamps, referenceTimestamps):
    """
    Corrects a numpy array of datetime64 timestamps if they are affected by daylight savings.
    This is a simple correction -- if there is an offset between 59 minutes and 61 minutes,
    the timestamps are assumed to be exactly one hour in the future.
    :param timestamps: Numpy array, dtype='datetime64[ms]'
    :param referenceTimestamps: Numpy array, dtype='datetime64[ms]'
    :return: timestamps: Numpy array, dtype='datetime64[ms]'
    """
    tZero = timestamps[0]
    dt1 = np.timedelta64(59, 'm')
    dt2 = np.timedelta64(61, 'm')
    for i in range(len(referenceTimestamps)):
        reftZero = referenceTimestamps[i]
        if (reftZero + dt1 < tZero) & (reftZero + dt2 > tZero):
            timestamps = timestamps - np.timedelta64(1, "h")
            break
    return timestamps


def create_empty_obssum_recarray(nentries):
    """
    Sets up the default recarray of each observation with nentries as the length of each field
    """

    starttime = np.zeros(nentries, dtype='datetime64[ms]')
    endtime = np.zeros(nentries, dtype='datetime64[ms]')
    slon = np.zeros(nentries, dtype=float)
    slat = np.zeros(nentries, dtype=float)
    duration = np.zeros(nentries, dtype='timedelta64[m]')
    rosaGBmean = np.zeros(nentries, dtype=float)
    rosaGBstd = np.zeros(nentries, dtype=float)
    rosaCTmean = np.zeros(nentries, dtype=float)
    rosaCTstd = np.zeros(nentries, dtype=float)
    rosaCKmean = np.zeros(nentries, dtype=float)
    rosaCKstd = np.zeros(nentries, dtype=float)
    zylaMean = np.zeros(nentries, dtype=float)
    zylaStd = np.zeros(nentries, dtype=float)
    ibisMean = np.zeros(nentries, dtype=float)
    ibisStd = np.zeros(nentries, dtype=float)
    scinMean = np.zeros(nentries, dtype=float)
    scinStd = np.zeros(nentries, dtype=float)

    # noinspection PyTypeChecker
    obsArray = np.rec.fromarrays(
        [
            starttime,
            endtime,
            slon,
            slat,
            duration,
            rosaGBmean,
            rosaGBstd,
            rosaCTmean,
            rosaCTstd,
            rosaCKmean,
            rosaCKstd,
            zylaMean,
            zylaStd,
            ibisMean,
            ibisStd,
            scinMean,
            scinStd
        ],
        names=[
            'start',
            'end',
            'slon',
            'slat',
            'duration',
            'RosaGBMean',
            'RosaGBStd',
            'RosaCTMean',
            'RosaCTStd',
            'RosaCKMean',
            'RosaCKStd',
            'zylaMean',
            'zylaStd',
            'ibisMean',
            'ibisStd',
            'scinMean',
            'scinStd'
        ]
    )
    return obsArray


def characterize_seeing_quality(baseDir, obsDate):
    """
    Sets up observing summary parameters for all instruments,
    and determines the base parameters needed for the quality metrics used.

    Rather than returning obssums for every instrument, it pares down the information.
    It returns a structure with the following data products:
        1.) Starttime
        2.) Endtime
        3.) Solar-Longitude
        4.) Solar-Latitude
        5.) Duration
        6.) Gband Mean HSM
        7.) Gband Std HSM
        8.) Continuum Mean HSM
        9.) Continuum Std HSM
        10.) CaK Mean HSM
        11.) CaK Std HSM
        12.) Zyla Mean MFGS
        13.) Zyla Std MFGS
        14.) IBIS Mean HSM
        15.) IBIS Std HSM
        16.) Scin Mean
        17.) Scin Std
    Each of these is an array with shape (N. Pointings). This is determined by the number
    of FIRS maps on that day. If there are no FIRS maps, the flow for determining pointings is:
        1.) FIRS
        2.) IBIS
        3.) SPINOR pointings with ROSA times
            a.) If no ROSA times, SPINOR pointings and times
        4.) HSG pointings with ROSA times
            a.) If no ROSA times, HSG obssums are combined into one
            This is because HSG has its fast raster mode.
        5.) ROSA/Zyla times with no pointings.
    """

    firsObssum = obssum.create_firs_obssum(baseDir, obsDate)
    # These three return their wavelengths as well.
    # Don't need em RN
    ibisObssum, _ = obssum.create_ibis_obssum(baseDir)
    spinorObssum, _ = obssum.create_spinor_obssum(baseDir, obsDate)
    hsgObssum, _ = obssum.create_hsg_obssum(baseDir, obsDate)

    rosaObssum, _ = obssum.create_rosa_obssum(baseDir, obsDate)
    try:
        zylaObssum = obssum.create_zyla_obssum(baseDir, obsDate)
    except (Exception,):
        zylaObssum = np.rec.fromarrays(
            [np.ones(0), np.ones(0), np.ones(0), np.ones(0)],
            names=['start', 'end', 'type', 'directory']
        )

    if len(firsObssum['start'][firsObssum['type'] == 'scan']) > 0:
        observationOverview = create_empty_obssum_recarray(
            len(firsObssum['start'][firsObssum['type'] == 'scan'])
        )
        observationOverview['start'] = firsObssum['start'][firsObssum['type'] == 'scan']
        observationOverview['end'] = firsObssum['end'][firsObssum['type'] == 'scan']
        observationOverview['duration'] = (
                observationOverview['end'] - observationOverview['start']
        ).astype('timedelta64[m]')
        observationOverview['scinMean'] = firsObssum['scin'][firsObssum['type'] == 'scan']
        observationOverview['scinStd'] = firsObssum['stdscin'][firsObssum['type'] == 'scan']
        observationOverview['slat'] = firsObssum['slat'][firsObssum['type'] == 'scan']
        observationOverview['slon'] = firsObssum['slon'][firsObssum['type'] == 'scan']
    elif len(ibisObssum['start'][ibisObssum['type'] == 'scan']) > 0:
        observationOverview = create_empty_obssum_recarray(
            len(ibisObssum['start'][ibisObssum['type'] == 'scan'])
        )
        observationOverview['start'] = ibisObssum['start'][ibisObssum['type'] == 'scan']
        observationOverview['end'] = ibisObssum['end'][ibisObssum['type'] == 'scan']
        observationOverview['duration'] = (
            observationOverview['end'] - observationOverview['start']
        ).astype('timedelta64[m]')
        observationOverview['scinMean'] = ibisObssum['scin'][ibisObssum['type'] == 'scan']
        observationOverview['scinStd'] = ibisObssum['stdscin'][ibisObssum['type'] == 'scan']
        observationOverview['slat'] = ibisObssum['slat'][ibisObssum['type'] == 'scan']
        observationOverview['slon'] = ibisObssum['slon'][ibisObssum['type'] == 'scan']
    elif len(spinorObssum['start'][spinorObssum['type'] == 'scan']) > 0:
        if len(rosaObssum['start'][rosaObssum['type'] == 'map']) > 0:
            observationOverview = create_empty_obssum_recarray(
                len(rosaObssum['start'][rosaObssum['type'] == 'map'])
            )
            starttimes = rosaObssum['start'][rosaObssum['type'] == 'map']
            endtimes = rosaObssum['end'][rosaObssum['type'] == 'map']
            slat = np.zeros(len(starttimes))
            slon = np.zeros(len(starttimes))
            scin = np.zeros(len(starttimes))
            stdscin = np.zeros(len(starttimes))
            for i in range(len(starttimes)):
                spinorIndex = _find_nearest(
                    spinorObssum['start'],
                    starttimes[i]
                )
                slat[i] = spinorObssum['slat'][spinorIndex]
                slon[i] = spinorObssum['slon'][spinorIndex]
                scin[i] = spinorObssum['scin'][spinorIndex]
                stdscin[i] = spinorObssum['stdscin'][spinorIndex]
        else:
            observationOverview = create_empty_obssum_recarray(
                len(spinorObssum['start'][spinorObssum['type'] == 'map'])
            )
            starttimes = spinorObssum['start'][spinorObssum['type'] == 'scan']
            endtimes = spinorObssum['end'][spinorObssum['type'] == 'scan']
            slat = spinorObssum['slat'][spinorObssum['type'] == 'scan']
            slon = spinorObssum['slon'][spinorObssum['type'] == 'scan']
            scin = spinorObssum['scin'][spinorObssum['type'] == 'scan']
            stdscin = spinorObssum['stdscin'][spinorObssum['type'] == 'scan']
        observationOverview['start'] = starttimes
        observationOverview['end'] = endtimes
        observationOverview['duration'] = (endtimes - starttimes).astype('timedelta64[m]')
        observationOverview['scinMean'] = scin
        observationOverview['scinStd'] = stdscin
        observationOverview['slon'] = slon
        observationOverview['slat'] = slat
    elif len(hsgObssum['start'][hsgObssum['type'] == 'scan']) > 0:
        if len(rosaObssum['start'][rosaObssum['type'] == 'map']) > 0:
            observationOverview = create_empty_obssum_recarray(
                len(rosaObssum['start'][rosaObssum['type'] == 'map'])
            )
            starttimes = rosaObssum['start'][rosaObssum['type'] == 'map']
            endtimes = rosaObssum['end'][rosaObssum['type'] == 'map']
            slat = np.zeros(len(starttimes))
            slon = np.zeros(len(starttimes))
            scin = np.zeros(len(starttimes))
            stdscin = np.zeros(len(starttimes))
            # Hypothetical: HSG run partially in fast raster during continuous
            # ROSA obs. Here, the pointing can just go to the 0th HSG file. Easy.
            # Scintillation should be done by np.average with weights by the duration
            # in seconds of the obs.
            if len(starttimes) == 1:
                slat[0] = hsgObssum['slat'][hsgObssum['type'] == 'scan'][0]
                slon[0] = hsgObssum['slon'][hsgObssum['type'] == 'scan'][0]
                hsgEnds = hsgObssum['end'][hsgObssum['type'] == 'scan']
                hsgStarts = hsgObssum['start'][hsgObssum['type'] == 'scan']
                hsgDurations = (hsgEnds - hsgStarts).astype('timedelta64[s]').astype(int)
                scin[0] = np.average(
                    hsgObssum['scin'][hsgObssum['type'] == 'scan'],
                    weights=hsgDurations
                )
                stdscin[0] = np.average(
                    hsgObssum['stdscin'][hsgObssum['type'] == 'scan'],
                    weights=hsgDurations
                )
            else:
                for i in range(len(starttimes)):
                    hsgStartIndex = _find_nearest(
                        hsgObssum['start'][hsgObssum['type'] == 'scan'],
                        starttimes[i]
                    )
                    hsgEndIndex = _find_nearest(
                        hsgObssum['end'][hsgObssum['type'] == 'scan'],
                        endtimes[i]
                    )
                    slat[i] = hsgObssum['slat'][hsgObssum['type'] == 'scan'][hsgStartIndex]
                    slon[i] = hsgObssum['slon'][hsgObssum['type'] == 'scan'][hsgStartIndex]
                    hsgEnds = hsgObssum['end'][hsgObssum['type'] == 'scan'][hsgStartIndex:hsgEndIndex]
                    hsgStarts = hsgObssum['start'][hsgObssum['type'] == 'scan'][hsgStartIndex:hsgEndIndex]
                    hsgDurations = (hsgEnds - hsgStarts).astype('timedelta64[s]').astype(int)
                    try:
                        scin[i] = np.average(
                            hsgObssum['scin'][hsgObssum['type'] == 'scan'][hsgStartIndex:hsgEndIndex],
                            weights=hsgDurations
                        )
                        stdscin[i] = np.average(
                            hsgObssum['stdscin'][hsgObssum['type'] == 'scan'][hsgStartIndex:hsgEndIndex],
                            weights=hsgDurations
                        )
                    except (Exception,):
                        scin[i] = None
                        stdscin[i] = None
        else:
            observationOverview = create_empty_obssum_recarray(1)
            starttimes = np.array([hsgObssum['start'][hsgObssum['type'] == 'scan'][0]])
            endtimes = np.array([hsgObssum['end'][hsgObssum['type'] == 'scan'][0]])
            slat = np.array([hsgObssum['slat'][hsgObssum['type'] == 'scan'][0]])
            slon = np.array([hsgObssum['slon'][hsgObssum['type'] == 'scan'][0]])
            hsgEnds = hsgObssum['end'][hsgObssum['type'] == 'scan']
            hsgStarts = hsgObssum['start'][hsgObssum['type'] == 'scan']
            hsgDurations = (hsgEnds - hsgStarts).astype('timedelta64[s]').astype(int)
            scin = np.array([np.average(
                hsgObssum['scin'][hsgObssum['type'] == 'scan'],
                weights=hsgDurations
            )])
            stdscin = np.array([np.average(
                hsgObssum['stdscin'][hsgObssum['type'] == 'scan'],
                weights=hsgDurations
            )])
        observationOverview['start'] = starttimes
        observationOverview['end'] = endtimes
        observationOverview['duration'] = (endtimes - starttimes).astype('timedelta64[m]')
        observationOverview['scinMean'] = scin
        observationOverview['scinStd'] = stdscin
        observationOverview['slon'] = slon
        observationOverview['slat'] = slat
    elif len(rosaObssum['start'][rosaObssum['type'] == 'map']) > 0:
        observationOverview = create_empty_obssum_recarray(
            len(rosaObssum['start'][rosaObssum['type'] == 'map'])
        )
        observationOverview['start'] = rosaObssum['start'][rosaObssum['type'] == 'map']
        observationOverview['end'] = rosaObssum['end'][rosaObssum['type'] == 'map']
        observationOverview['duration'] = (
            observationOverview['start'] - observationOverview['end']
        ).astype('timedelta64[m]')
        observationOverview['slon'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['slat'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['scinMean'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['scinStd'] = np.ones(len(observationOverview['start'])) * np.nan
    elif len(zylaObssum['start'][zylaObssum['type'] == 'map']) > 0:
        observationOverview = create_empty_obssum_recarray(
            len(zylaObssum['start'][zylaObssum['type'] == 'map'])
        )
        observationOverview['start'] = zylaObssum['start'][zylaObssum['type'] == 'map']
        observationOverview['end'] = zylaObssum['end'][zylaObssum['type'] == 'map']
        observationOverview['duration'] = (
                observationOverview['start'] - observationOverview['end']
        ).astype('timedelta64[m]')
        observationOverview['slon'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['slat'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['scinMean'] = np.ones(len(observationOverview['start'])) * np.nan
        observationOverview['scinStd'] = np.ones(len(observationOverview['start'])) * np.nan
    else:
        # Null case: no usable obssums found
        # Mostly, these will be polcal days.
        observationOverview = create_empty_obssum_recarray(0)
        return observationOverview

    rosaSeeingFiles, _ = seeing.rosa_seeing_quality(baseDir)
    zylaSeeingFiles, _ = seeing.zyla_seeing_quality(baseDir, obsDate)
    ibisSeeingFiles, _ = seeing.ibis_seeing_quality(baseDir)

    # If there are differing amounts of these, something is fuqd in the archive.
    # There's a contingency I could code for here, but I'm already behind schedule.
    gbandSeeingFiles = [x for x in rosaSeeingFiles if 'gband' in x]
    cakSeeingFiles = [x for x in rosaSeeingFiles if 'cak' in x]
    contSeeingFiles = [x for x in rosaSeeingFiles if '4170' in x]
    # I spent a while standardizing the various seeing parameter files.
    # I'm welcome. It was hell.
    # How do we do this?
    # Fine-grained. We split rosa into CaK, gband, 4170
    # Write a function that checks CaK for daylight savings weirdness
    # against a reference by looking if the starttime is between 30-90 minutes ahead
    # Check if there are as many obs as seeing files.
    # If there are fewer obs than seeing files, we check start of each seeing file against start obs.
    # Fill by weighted mean of seeing
    # If there are more obs than seeing files, fill by nearest as in base case

    rosa_len = np.max([len(gbandSeeingFiles), len(cakSeeingFiles), len(contSeeingFiles)])
    # Easy case -- same number of seeing params as obs in series. Can assume 1-1.
    # If there are fewer seeing params, then we just duplicate and fill.
    if rosa_len == len(observationOverview['start']):
        gbStarts = []
        for file in gbandSeeingFiles:
            see = np.load(file)
            obsIndex = _find_nearest(observationOverview['start'], see['TIMESTAMPS'][0])
            observationOverview['RosaGBMean'][obsIndex] = np.nanmean(see['HSM'])
            observationOverview['RosaGBStd'][obsIndex] = np.nanstd(see['HSM'])
            gbStarts.append(see['TIMESTAMPS'][0])
        contStarts = []
        for file in contSeeingFiles:
            see = np.load(file)
            obsIndex = _find_nearest(observationOverview['start'], see['TIMESTAMPS'][0])
            observationOverview['RosaCTMean'][obsIndex] = np.nanmean(see['HSM'])
            observationOverview['RosaCTStd'][obsIndex] = np.nanstd(see['HSM'])
            contStarts.append(see['TIMESTAMPS'][0])
        if len(gbStarts) >= len(contStarts):
            referenceStarttimes = gbStarts
        else:
            referenceStarttimes = contStarts
        for i in range(len(cakSeeingFiles)):
            see = np.load(cakSeeingFiles[i])
            corrTs = _correct_daylight_savings(see['TIMESTAMPS'], referenceStarttimes)
            obsIndex = _find_nearest(observationOverview['start'], corrTs[0])
            observationOverview['RosaCKMean'][obsIndex] = np.nanmean(see['HSM'])
            observationOverview['RosaCKStd'][obsIndex] = np.nanstd(see['HSM'])
    elif rosa_len < len(observationOverview['start']):
        weights = []
        mean_hsm = []
        std_hsm = []
        for file in gbandSeeingFiles:
            see = np.load(file)
            weights.append(len(see['HSM']))
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        if len(weights) != 0:
            for i in range(len(observationOverview['start'])):
                observationOverview['RosaGBMean'][i] = np.average(mean_hsm, weights=weights)
                observationOverview['RosaGBStd'][i] = np.average(std_hsm, weights=weights)
        weights = []
        mean_hsm = []
        std_hsm = []
        for file in contSeeingFiles:
            see = np.load(file)
            weights.append(len(see['HSM']))
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        if len(weights) != 0:
            for i in range(len(observationOverview['start'])):
                observationOverview['RosaCTMean'][i] = np.average(mean_hsm, weights=weights)
                observationOverview['RosaCTStd'][i] = np.average(std_hsm, weights=weights)
        weights = []
        mean_hsm = []
        std_hsm = []
        for file in cakSeeingFiles:
            see = np.load(file)
            weights.append(len(see['HSM']))
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        if len(weights) != 0:
            for i in range(len(observationOverview['start'])):
                observationOverview['RosaCKMean'][i] = np.average(mean_hsm, weights=weights)
                observationOverview['RosaCKStd'][i] = np.average(std_hsm, weights=weights)
    # Harder case -- more ROSA obs than base. Gotta combine.
    else:
        # GBand
        durs = []
        obsIndex = []
        mean_hsm = []
        std_hsm = []
        gband_starts = []
        for file in gbandSeeingFiles:
            see = np.load(file)
            obsIndex.append(_find_nearest(observationOverview['start'], see['TIMESTAMPS'][0]))
            durs.append(see['TIMESTAMPS'][-1] - see['TIMESTAMPS'][0])
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
            gband_starts.append(see['TIMESTAMPS'][0])
        indices = np.array(list(set(obsIndex)))
        obsIndex = np.array(obsIndex)
        durs = np.array(durs).astype('timedelta64[m]').astype(int)
        mean_hsm = np.array(mean_hsm)
        std_hsm = np.array(mean_hsm)
        for i in range(len(indices)):
            selection = obsIndex == indices[i]
            dur_select = durs[selection]
            mean_select = mean_hsm[selection]
            std_select = std_hsm[selection]
            observationOverview['RosaGBMean'][indices[i]] = np.average(mean_select, weights=dur_select)
            observationOverview['RosaGBStd'][indices[i]] = np.average(std_select, weights=dur_select)
        # Continuum
        durs = []
        obsIndex = []
        mean_hsm = []
        std_hsm = []
        cont_starts = []
        for file in contSeeingFiles:
            see = np.load(file)
            obsIndex.append(_find_nearest(observationOverview['start'], see['TIMESTAMPS'][0]))
            durs.append(see['TIMESTAMPS'][-1] - see['TIMESTAMPS'][0])
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
            cont_starts.append(see['TIMESTAMPS'][0])
        indices = np.array(list(set(obsIndex)))
        obsIndex = np.array(obsIndex)
        durs = np.array(durs).astype('timedelta64[m]').astype(int)
        mean_hsm = np.array(mean_hsm)
        std_hsm = np.array(mean_hsm)
        for i in range(len(indices)):
            selection = obsIndex == indices[i]
            dur_select = durs[selection]
            mean_select = mean_hsm[selection]
            std_select = std_hsm[selection]
            observationOverview['RosaCTMean'][indices[i]] = np.average(mean_select, weights=dur_select)
            observationOverview['RosaCTStd'][indices[i]] = np.average(std_select, weights=dur_select)
        # Ca K
        durs = []
        obsIndex = []
        mean_hsm = []
        std_hsm = []
        if len(gband_starts) >= len(cont_starts):
            reference_starts = gband_starts
        else:
            reference_starts = cont_starts
        for i in range(len(cakSeeingFiles)):
            see = np.load(cakSeeingFiles[i])
            corrTs = _correct_daylight_savings(see['TIMESTAMPS'], reference_starts)
            obsIndex.append(_find_nearest(observationOverview['start'], corrTs[0]))
            durs.append(corrTs[-1] - corrTs[0])
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        indices = np.array(list(set(obsIndex)))
        obsIndex = np.array(obsIndex)
        durs = np.array(durs).astype('timedelta64[m]').astype(int)
        mean_hsm = np.array(mean_hsm)
        std_hsm = np.array(mean_hsm)
        for i in range(len(indices)):
            selection = obsIndex == indices[i]
            dur_select = durs[selection]
            mean_select = mean_hsm[selection]
            std_select = std_hsm[selection]
            if np.nansum(dur_select) <= 0:
                continue
            observationOverview['RosaCKMean'][indices[i]] = np.average(mean_select, weights=dur_select)
            observationOverview['RosaCKStd'][indices[i]] = np.average(std_select, weights=dur_select)
    # The above should really be a function.
    # I know, but I'm gonna need you to get alllllll the way off my back about that.
    # Especially because we're gonna use the same logic for ibis as well...
    if len(ibisSeeingFiles) == len(observationOverview['start']):
        for file in ibisSeeingFiles:
            see = np.load(file)
            obsIndex = _find_nearest(observationOverview['start'], see['TIMESTAMPS'][0])
            observationOverview['ibisMean'][obsIndex] = np.nanmean(see['HSM'])
            observationOverview['ibisStd'][obsIndex] = np.nanstd(see['HSM'])
    # Fewer: Fill everything same value..
    elif len(ibisSeeingFiles) < len(observationOverview['start']):
        weights = []
        mean_hsm = []
        std_hsm = []
        for file in ibisSeeingFiles:
            see = np.load(file)
            weights.append(len(see['HSM']))
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        if len(weights) != 0:
            for i in range(len(observationOverview['start'])):
                observationOverview['ibisMean'][i] = np.average(mean_hsm, weights=weights)
                observationOverview['ibisStd'][i] = np.average(std_hsm, weights=weights)
    # Harder case -- more IBIS obs than base. Gotta combine.
    else:
        durs = []
        obsIndex = []
        mean_hsm = []
        std_hsm = []
        for file in ibisSeeingFiles:
            see = np.load(file)
            obsIndex.append(_find_nearest(observationOverview['start'], see['TIMESTAMPS'][0]))
            durs.append(see['TIMESTAMPS'][-1] - see['TIMESTAMPS'][0])
            mean_hsm.append(np.nanmean(see['HSM']))
            std_hsm.append(np.nanstd(see['HSM']))
        indices = np.array(list(set(obsIndex)))
        obsIndex = np.array(obsIndex)
        durs = np.array(durs).astype('timedelta64[m]').astype(int)
        mean_hsm = np.array(mean_hsm)
        std_hsm = np.array(mean_hsm)
        for i in range(len(indices)):
            selection = obsIndex == indices[i]
            dur_select = durs[selection]
            mean_select = mean_hsm[selection]
            std_select = std_hsm[selection]
            observationOverview['ibisMean'][indices[i]] = np.average(mean_select, weights=dur_select)
            observationOverview['ibisStd'][indices[i]] = np.average(std_select, weights=dur_select)

    # Now Zyla, which is more of a problem. Again, no timestamps.
    # If there are equal or fewer Zyla seeing files, we can do the naive thing again.
    # Only this time, if there are fewer, we fill every one with the weighted mean.
    if len(zylaSeeingFiles) == len(observationOverview['start']):
        for i in range(len(zylaSeeingFiles)):
            see = np.load(zylaSeeingFiles[i])
            observationOverview['zylaMean'][i] = np.nanmean(see['MFGS'])
            observationOverview['zylaStd'][i] = np.nanstd(see['MFGS'])
    # Fill everything with the same value
    elif len(zylaSeeingFiles) < len(observationOverview['start']):
        weights = []
        mean_mfgs = []
        std_mfgs = []
        for file in zylaSeeingFiles:
            see = np.load(file)
            weights.append(len(see['MFGS']))
            mean_mfgs.append(np.nanmean(see['MFGS']))
            std_mfgs.append(np.nanstd(see['MFGS']))
        if len(weights) != 0:
            for i in range(len(observationOverview['start'])):
                observationOverview['zylaMean'][i] = np.average(mean_mfgs, weights=weights)
                observationOverview['zylaStd'][i] = np.average(std_mfgs, weights=weights)
    # Hard Mode. More Zyla files than obs. Things need to be combined, but what?
    # If the Zyla obssum has the same number of map entries as there are seeing files,
    # We can pick the starttimes out from that. If there are no starttimes, or if there
    # Are a different number of map entries, we fall back to the previous case.
    else:
        if len(zylaObssum['start'][zylaObssum['type'] == 'map']) > 0:
            if (
                    (len(zylaSeeingFiles) == len(zylaObssum['start'][zylaObssum['type'] == 'map'])) &
                    (not np.isnan(zylaObssum['start'][zylaObssum['type'] == 'map'][0]))
            ):
                weights = []
                obsIndex = []
                mean_mfgs = []
                std_mfgs = []
                for i in range(len(zylaSeeingFiles)):
                    see = np.load(zylaSeeingFiles[i])
                    obsIndex.append(
                        _find_nearest(
                            observationOverview['start'],
                            zylaObssum['start'][zylaObssum['type'] == 'map'][i]
                        )
                    )
                    weights.append(len(see['MFGS']))
                    mean_mfgs.append(np.nanmean(see['MFGS']))
                    std_mfgs.append(np.nanstd(see['MFGS']))
                indices = np.array(list(set(obsIndex)))
                obsIndex = np.array(obsIndex)
                weights = np.array(weights)
                mean_mfgs = np.array(mean_mfgs)
                std_mfgs = np.array(mean_mfgs)
                for i in range(len(indices)):
                    selection = obsIndex == indices[i]
                    weights_select = weights[selection]
                    mean_select = mean_mfgs[selection]
                    std_select = std_mfgs[selection]
                    observationOverview['zylaMean'][indices[i]] = np.average(mean_select, weights=weights_select)
                    observationOverview['zylaStd'][indices[i]] = np.average(std_select, weights=weights_select)
        else:
            weights = []
            mean_mfgs = []
            std_mfgs = []
            for file in zylaSeeingFiles:
                see = np.load(file)
                weights.append(len(see['MFGS']))
                mean_mfgs.append(np.nanmean(see['MFGS']))
                std_mfgs.append(np.nanstd(see['MFGS']))
            if len(weights) != 0:
                for i in range(len(observationOverview['start'])):
                    observationOverview['zylaMean'][i] = np.average(mean_mfgs, weights=weights)
                    observationOverview['zylaStd'][i] = np.average(std_mfgs, weights=weights)
    return observationOverview
