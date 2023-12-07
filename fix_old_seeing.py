"""
Alright. All the ROSA seeing files that were generated with older versions of the seeing
parameterization code are effed up. I tried finding gaps in the timestamps rather than
creating new files per time series. As a result, everything is effed.

We need to iterate over the archive and
1.) Find the ROSA seeing param files
    a.) Fortunately, Zyla/IBIS should be fine
2.) Find the corresponding ROSA data directory
3.) Split the files in the data directory by observing series
    a.) No flat/darks, only series with more than 5 files in the series
4.) Check the start and end times of those series.
5.) Find the corresponding timestamps in the ROSA seeing param file
    a.) Find_nearest will work, but do +1 to start and -1 to end.
6.) Split the arrays in these files along those lines.
6.5.) Remove the old files
7.) Save the new files
8.) While we're at it, older params had save files for extensions per file instead
    of having the reference images saved directly to a npy save file.
    So write the reference images files.
    In fact, you have to do this for every series! Cause those won't be correct either
    Fuck.
    Okay.
    We can check the number of extensions in the last file per series, and assume the
    standard 256 otherwise.
8.5.) Remove the old refim files
9.) Write the new reference image files.
10.) Remake the plots.
11.) Re-generate the webpage with the new files.
Dammit.
"""

import numpy as np
import astropy.io.fits as fits
import tqdm
import glob
import os
import create_plots as cplt
import generate_webpage as gw


def _find_nearest(array, value):
    """Determines the index of the closest value in an array to a sepecified other value

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


# basedirs = sorted(glob.glob("/sunspot/solardata/20*/*/*/"))[81:]

basedirs = ["/sunspot/solardata/2019/04/08"]

filters = ['4170', 'gband', 'cak', '3500']
for i in tqdm.tqdm(range(len(basedirs))):
    print(basedirs[i])
    for band in filters:
        seeingFiles = sorted(
            glob.glob(
                os.path.join(
                    basedirs[i],
                    '*'+band+'_seeing_quality*'
                )
            )
        )
        referenceImages = sorted(
            glob.glob(
                os.path.join(
                    basedirs[i],
                    '*'+band+'_key_images*'
                )
            )
        )
        imsPerFile = sorted(
            glob.glob(
                os.path.join(
                    basedirs[i],
                    '*'+band+'_imsperfile*'
                )
            )
        )
        plots = sorted(
            glob.glob(
                os.path.join(
                    basedirs[i],
                    '*'+band+'*.png'
                )
            )
        )
        if len(seeingFiles) == 0:
            continue
        # There's a lot of things in this world you're gonna have absolutely no use for.
        # Seeing parameters from the 3500 band are among these things.
        if band == '3500':
            for file in imsPerFile:
                os.remove(file)
            for file in seeingFiles:
                os.remove(file)
            for file in plots:
                os.remove(file)
            for file in referenceImages:
                os.remove(file)
            continue
        level0Directory = glob.glob(
            os.path.join(
                basedirs[i],
                '**',
                'level0',
                '**',
                '*'+band+'*'
            ), recursive=True
        )
        if len(level0Directory) == 0:
            continue
        level0Directory = level0Directory[0]
        obsSeries = [
            series for series in set(
                [
                    "_".join(z.split("_")[:-1]) for z in
                    [
                        q for q in glob.glob(
                            os.path.join(level0Directory, '*fit*')
                        ) if 'dark' not in q and 'flat' not in q
                    ]
                ]
            ) if len(glob.glob(series + "*fit*")) > 10
        ]

        seriesStarttimes = []
        seriesEndtimes = []
        for series in obsSeries:
            flist = sorted(glob.glob(series + '*fit*'))
            with fits.open(flist[0]) as f:
                seriesStarttimes.append(
                    np.datetime64(f[1].header['DATE'])
                )
            with fits.open(flist[-1]) as f:
                dframe = int(f[0].header['HIERARCH FRAME RATE'].split(" ")[0])
                td = np.timedelta64(
                    int(
                        (len(f) - 1) * dframe
                    ),
                    'ms'
                )
                seriesEndtimes.append(
                    np.datetime64(f[1].header['DATE']) + td
                )
        timestamps = []
        hsm = []
        central_hsm = []
        central_pix = []
        central_mean = []
        image_mean = []
        image_rms = []
        central_rms = []

        for file in seeingFiles:
            try:
                seeingParams = np.load(file)
                time_key = 'TIMESTAMPS'
                hsm_key = 'HSM'
                hsm_center_key = 'CENTRAL_HSM'
                cpix_key = 'CENTRAL_PIXEL'
                csub_key = 'CENTRAL_MEAN'
                immn_key = 'IMAGE_MEAN'
                rms_key = 'IMAGE_RMS'
                crms_key = 'CENTRAL_RMS'
            except (Exception,):
                seeingParams = np.load(file, allow_pickle=True).flat[0]
                time_key = 'Timestamp'
                hsm_key = 'HSM'
                hsm_center_key = 'Central HSM'
                cpix_key = 'Central Pixel Value'
                csub_key = 'Sum of Image Center'
                immn_key = 'Image Mean Value'
                rms_key = 'RMS of Image'
                crms_key = 'RMS of Image Center'
            timestamps += [k for k in seeingParams[time_key]]
            hsm += [k for k in seeingParams[hsm_key]]
            central_hsm += [k for k in seeingParams[hsm_center_key]]
            try:
                central_pix += [k for k in seeingParams[cpix_key]]
            except (Exception,):
                central_pix += [k for k in seeingParams['Cental Pixel Value']]
            central_mean += [k for k in seeingParams[csub_key]]
            image_mean += [k for k in seeingParams[immn_key]]
            image_rms += [k for k in seeingParams[rms_key]]
            central_rms += [k for k in seeingParams[crms_key]]
        timestamps = np.array(timestamps)
        if timestamps.dtype == '<U32':
            tcut = timestamps != 'nan'
        else:
            tcut = np.ones(len(timestamps), dtype=bool)

        timestamps = np.array(timestamps[tcut], dtype='datetime64[ms]')
        hsm = np.array(hsm, dtype=float)
        central_hsm = np.array(central_hsm, dtype=float)
        central_pix = np.array(central_pix, dtype=float)
        central_mean = np.array(central_mean, dtype=float)
        image_mean = np.array(image_mean, dtype=float)
        image_rms = np.array(image_rms, dtype=float)
        central_rms = np.array(central_rms, dtype=float)
        listofparams = [
            hsm, central_hsm, central_pix,
            central_mean, image_mean, image_rms,
            central_rms
        ]
        lens = []
        for p in listofparams:
            lens.append(len(p))
        shortest = lens[lens.index(np.min(lens))]
        tcut = tcut[:shortest-1]
        timestamps = timestamps[:shortest-1]
        hsm = hsm[:shortest-1][tcut]
        central_hsm = central_hsm[:shortest-1][tcut]
        central_pix = central_pix[:shortest-1][tcut]
        image_mean = image_mean[:shortest-1][tcut]
        image_rms = image_rms[:shortest-1][tcut]
        central_rms = central_rms[:shortest-1][tcut]

        newParams = []
        newRefs = []
        for j in range(len(seriesStarttimes)):
            startIdx = _find_nearest(timestamps, seriesStarttimes[j])
            endIdx = _find_nearest(timestamps, seriesEndtimes[j])
            if startIdx == endIdx:
                continue
            obsDate = obsSeries[j].split("_")[-2].replace("-", "")
            obsTime = obsSeries[j].split("_")[-1].replace(":", "")
            seeingSaveStr = os.path.join(basedirs[i], obsDate + "_" + obsTime + "_" + band + "_seeing_quality.npy")
            refimSaveStr = os.path.join(basedirs[i], obsDate + "_" + obsTime + "_" + band + "_key_images.npy")
            seriesFiles = sorted(glob.glob(obsSeries[j] + '*fit*'))
            seeingArray = np.rec.fromarrays(
                [
                    timestamps[startIdx:endIdx],
                    hsm[startIdx:endIdx],
                    central_hsm[startIdx:endIdx],
                    central_pix[startIdx:endIdx],
                    central_mean[startIdx:endIdx],
                    image_mean[startIdx:endIdx],
                    image_rms[startIdx:endIdx],
                    central_rms[startIdx:endIdx]
                ],
                names=[
                    'TIMESTAMPS',
                    'HSM',
                    'CENTRAL_HSM',
                    'CENTRAL_PIXEL',
                    'CENTRAL_MEAN',
                    'IMAGE_MEAN',
                    'IMAGE_RMS',
                    'CENTRAL_RMS'
                ]
            )
            try:
                medianIndex = _find_nearest(np.nan_to_num(seeingArray['HSM']), np.nanmedian(seeingArray['HSM']))
                medFile = int(medianIndex/256)
                medExt = medianIndex % 256
                medianImage = fits.open(seriesFiles[medFile])[medExt+1].data

                minIndex = _find_nearest(
                    np.nan_to_num(seeingArray['HSM']),
                    np.nanmean(seeingArray['HSM']) - np.nanstd(seeingArray['HSM'])
                )
                minFile = int(minIndex / 256)
                minExt = minIndex % 256
                minImage = fits.open(seriesFiles[minFile])[minExt+1].data

                maxIndex = _find_nearest(
                    np.nan_to_num(seeingArray['HSM']),
                    np.nanmean(seeingArray['HSM']) + np.nanstd(seeingArray['HSM'])
                )
                maxFile = int(maxIndex / 256)
                maxExt = maxIndex % 256
                maxImage = fits.open(seriesFiles[maxFile])[maxExt+1].data
            except (Exception,):
                print(Exception)
                print("Probably a file extension mismatch...")
                exts = []
                for fitsfile in seriesFiles:
                    exts.append(len(fits.open(fitsfile))-1)
                medianIndex = _find_nearest(
                    np.nan_to_num(seeingArray['HSM']),
                    np.nanmedian(seeingArray['HSM'])
                )
                ctr = 0
                imgIndex = 0
                while ctr < medianIndex:
                    ctr += exts[imgIndex]
                    imgIndex += 1
                medianImage = fits.open(
                    seriesFiles[imgIndex - 1]
                )[medianIndex - (ctr - exts[imgIndex - 1])].data

                minIndex = _find_nearest(
                    np.nan_to_num(seeingArray['HSM']),
                    np.nanmean(seeingArray['HSM']) - np.nanstd(seeingArray['HSM'])
                )
                ctr = 0
                imgIndex = 0
                while ctr < minIndex:
                    ctr += exts[imgIndex]
                    imgIndex += 1
                minImage = fits.open(
                    seriesFiles[imgIndex - 1]
                )[minIndex - (ctr - exts[imgIndex - 1])].data

                maxIndex = _find_nearest(
                    np.nan_to_num(seeingArray['HSM']),
                    np.nanmean(seeingArray['HSM']) + np.nanstd(seeingArray['HSM'])
                )
                ctr = 0
                imgIndex = 0
                while ctr < maxIndex:
                    ctr += exts[imgIndex]
                    imgIndex += 1
                maxImage = fits.open(
                    seriesFiles[imgIndex - 1]
                )[maxIndex - (ctr - exts[imgIndex - 1])].data

            keyImages = np.rec.fromarrays(
                [
                    medianImage,
                    minImage,
                    maxImage
                ],
                names=[
                    'MEDIAN_IMAGE',
                    'MIN_IMAGE',
                    'MAX_IMAGE'
                ]
            )

            newParams.append(seeingSaveStr)
            newRefs.append(refimSaveStr)
            np.save(seeingSaveStr, seeingArray)
            np.save(refimSaveStr, keyImages)

        for file in plots:
            os.remove(file)
        for j in range(len(newParams)):
            cplt.create_seeing_plot(basedirs[i], newParams[j], newRefs[j], 'ROSA_'+band.upper())

        for file in imsPerFile:
            os.remove(file)
        for file in seeingFiles:
            if file not in newParams:
                os.remove(file)
        for file in referenceImages:
            if file not in newRefs:
                os.remove(file)

    gw.automatic(basedirs[i])
