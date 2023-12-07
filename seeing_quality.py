"""
Series of functions for the goodness of imaging data independant of scintillation measures:
These functions are the source of the files that are used to determine the coefficient of variation
Which is in turn used to determine the data quality.

For anyone in the future looking at this code:
    a.) I'm sorry.
    b.) The Dunn is equipped with a Seykora scintillation monitor, which basically moniters
        turbulence at the ~70m level. Most turbulence is at this level, but this is not
        particularly useful. For one, we have an AO system, which can return better/worse
        corrections depending on the obs target, operator, room temperature, when it was last
        aligned, etc. All the below determine the quality through various contrast metrics,
        which are AO-perturbed, and therefore, the best indicator of actually how good the data
        are. I don't use RMS, which is the typical, because I'm a hipster, I guess.
        It's pretty affected by light level. Instead, I use the Helmli-Scherer mean for continuum/Ca K
        data. And the Median Filter Gradiant Similarity metric for h-alpha data. HSM doesn't do well on
        extended chromospheric structures. MFGS is slightly better, but it is MUCH slower to compute.
        I've tried a lot of things to speed it up. This is about as fast as it gets, trust me.

        Dask arrays, other median filters, windowing, and a dozen other things, this is the fastest.
"""


import numpy as np
import glob
import dask
from dask.diagnostics import ProgressBar
import os
from astropy.io import fits
import warnings
from . import observation_summary as obssum

import zyla_tools as zt

from scipy.ndimage import uniform_filter, median_filter


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


def _hs_mean(frame, window_size=21):
    """Compute the Helmli-Scherer mean for a given image frame. Lower values denote "better" seeing.
    Parameters:
    -----------
    frame : array-like
        The image used for computation of the HS mean
    window_size : int
        Size of the window used in creating the mean filtered image.
        Passes through to scipy.ndimage.uniform_filter.
        I believe it can only be odd values, and may be a sequence if you'd like the box to not be a square.
    Returns:
    --------
    hs_mean : a single value describing the level of resolved structure in the image.
    """

    med_filt_frame = uniform_filter(frame, size=window_size)

    FM = med_filt_frame / frame

    idx_FM = (med_filt_frame < frame)

    FM[idx_FM] = frame[idx_FM]/med_filt_frame[idx_FM]

    hs_mean = np.mean(1./FM)

    return hs_mean


def _mfgs(image, kernel_size=3):
    """
    Median-filter gradient similarity metric from Deng et al., 2015
    Parameters:
    -----------
    image : array-like
        Numpy array with image data
    kernel_size : int
        smoothing window for median filtering

    Returns:
    --------
    mfgs_metric : float
        Image quality assesment
    """

    med_img = median_filter(image, size=kernel_size)
    grad_img = np.gradient(image)

    grad = (np.sum(np.abs(grad_img[0][image != 0])) + np.sum(np.abs(grad_img[1][image != 0])))
    med_grad = np.sum(np.abs(np.gradient(med_img)))

    mfgs_metric = ((2 * med_grad * grad) /
                   (med_grad**2 + grad**2))

    return mfgs_metric


@dask.delayed
def _rosa_params(fname):
    """
    Dask delayed function for determining seeing parameters from a ROSA fits file.
    Opens all image extensions and determines the:
    1.) Number of image extensions
    2.) Timestamps of each extension
    3.) Helmli-Scherer Mean of the image
    4.) Helmli-Scherer Mean of the central 100x100 pixels
    5.) The light curve of the central pixel
    6.) The lightcurve of the central 100x100 pixels
    7.) The lightcurve of the full frame
    8.) The RMS of the image
    9.) The RMS of the central 100x100 pixels

    Parameters:
    -----------
    fname : str
        Filename to analyze

    Returns
    -------
    seeingParameters : list
        List of lists of above params
    """
    file = fits.open(fname)
    try:
        exts = len(file) - 1
    except (Exception,):
        warnings.warn("Critically fucked ROSA File: " + fname)
        return [
            0,
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan]
        ]
    timestamps = []
    hsm = np.zeros(exts)
    hsmCenter = np.zeros(exts)
    centralPixel = np.zeros(exts)
    centralSquare = np.zeros(exts)
    frameMean = np.zeros(exts)
    frameRMS = np.zeros(exts)
    centralRMS = np.zeros(exts)
    for i in range(1, len(file)):
        image = file[i].data / float(file[0].header['EXPOSURE'][:-4])
        imageCenter = image[
                      int(image.shape[0]/2 - 50):int(image.shape[0]/2 + 50),
                      int(image.shape[1]/2 - 50):int(image.shape[1]/2 + 50)
                      ]
        try:
            timestamps.append(file[i].header['DATE'])
            hsm[i-1] = _hs_mean(image)
            hsmCenter[i-1] = _hs_mean(imageCenter)
            centralPixel[i-1] = image[int(image.shape[0]/2), int(image.shape[1]/2)]
            centralSquare[i-1] = np.nanmean(imageCenter)
            frameMean[i-1] = np.nanmean(image)
            frameRMS[i-1] = np.sum((image - frameMean[i-1])**2)/(image.shape[0]*image.shape[1])
            centralRMS[i-1] = np.sum((imageCenter - centralSquare[i-1])**2)/(imageCenter.shape[0]*imageCenter.shape[1])
        except (Exception,):
            timestamps.append(np.nan)
            hsm[i-1] = np.nan
            hsmCenter[i-1] = np.nan
            centralPixel[i-1] = np.nan
            centralSquare[i-1] = np.nan
            frameMean[i-1] = np.nan
            frameRMS[i-1] = np.nan
            centralRMS[i-1] = np.nan
    seeingParameters = [
        exts,
        timestamps,
        hsm,
        hsmCenter,
        centralPixel,
        centralSquare,
        frameMean,
        frameRMS,
        centralRMS
    ]
    return seeingParameters


@dask.delayed
def _ibis_params(fname):
    """
    Dask delayed function for determining seeing parameters from an IBIS whitelight fits file.
    Opens all image extensions and determines the:
    1.) Number of image extensions
    2.) Timestamps of each extension
    3.) Helmli-Scherer Mean of the image
    5.) The light curve of the central pixel
    6.) The lightcurve of the central 100x100 pixels
    7.) The lightcurve of the full frame
    8.) The RMS of the image

    Parameters:
    -----------
    fname : str
        Filename to analyze

    Returns
    -------
    seeingParameters : list
        List of lists of above params
    """
    try:
        file = fits.open(fname)
    except (Exception,):
        return [0, [], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
    exts = len(file) - 1
    timestamps = []
    hsm = np.zeros(exts)
    centralPixel = np.zeros(exts)
    centralSquare = np.zeros(exts)
    frameMean = np.zeros(exts)
    frameRMS = np.zeros(exts)
    for i in range(1, len(file)):
        try:
            # Kludge to avoid the circular ibis mask.
            # Fucks up spectropolarimetric data
            # This is rare in the archive, and I'm fine with not considering it.
            image = file[i].data[200:800, 200:800] / float(file[i].header['EXPTIME'])
            timestamps.append(file[i].header['DATE-OBS'])
            hsm[i-1] = _hs_mean(image)
            centralPixel[i-1] = image[300, 300]
            centralSquare[i-1] = np.nanmean(image[250:350, 250:350])
            frameMean[i-1] = np.nanmean(image)
            frameRMS = np.sum((image - frameMean[i-1])**2)/(600*600)
        except (Exception,):
            timestamps.append(np.nan)
            hsm[i-1] = np.nan
            centralPixel[i-1] = np.nan
            centralSquare[i-1] = np.nan
            frameMean[i-1] = np.nan
            frameRMS[i-1] = np.nan
    seeingParameters = [
        exts,
        timestamps,
        hsm,
        centralPixel,
        centralSquare,
        frameMean,
        frameRMS
    ]
    return seeingParameters


@dask.delayed
def _zyla_params(fname, zylaShape=None):
    """
    Dask delayed function for determining seeing parameters from Zyla Halpha images.
    Opens the image according to the extension and deermines:
    1.) Median Filter Gradiant Similarity of the image
    2.) The light curve of the central pixel
    3.) The lightcurve of the central 200x200 pixels
    4.) The lightcurve of the full frame
    5.) The RMS of the image
    6.) The RMS of the central 200x200 pixels

    Parameters:
    -----------
    fname : str
        Filename to analyze

    Returns
    -------
    seeingParameters : list
        List of lists of above params with fname leading because we have to resort after dask computing.
    """
    try:
        if "fits" in fname:
            image = fits.open(fname)[1].data
        else:
            image = zt.read_zyla(fname, dataShape=zylaShape, imageShape=zylaShape)
        imageCenter = image[
                      int(image.shape[0] / 2 - 100):int(image.shape[0] / 2 + 100),
                      int(image.shape[1] / 2 - 100):int(image.shape[1] / 2 + 100)]
        mfgs = _mfgs(image)
        centralPixel = image[int(image.shape[0]/2), int(image.shape[1]/2)]
        centralSquare = np.nanmean(imageCenter)
        frameMean = np.nanmean(image)
        frameRMS = np.nansum((image-frameMean)**2)/(image.shape[0] * image.shape[1])
        centralRMS = np.nansum((imageCenter-centralSquare)**2)/(imageCenter.shape[0]*imageCenter.shape[1])
    except (Exception,):
        mfgs = np.nan
        centralPixel = np.nan
        centralSquare = np.nan
        frameMean = np.nan
        frameRMS = np.nan
        centralRMS = np.nan
    seeingParameters = [
        fname,
        mfgs,
        centralPixel,
        centralSquare,
        frameMean,
        frameRMS,
        centralRMS
    ]
    return seeingParameters


def rosa_seeing_quality(baseDir):
    """
    Searches for ROSA seeing parameter and reference image files.
    If it can find them, returns the list of each type.
    Otherwise, searches for ROSA data and files,
    then determines seeing parameters for them, creates those files, and saves them.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD

    Returns:
    --------
    seeingFiles : list
        List of all rosa seeing parameter files found
    referenceImages : list
        List of files containing ROSA reference images
    """
    seeingFiles = []
    referenceImages = []
    rosaFilter = ['4170', 'gband', 'cak']
    for band in rosaFilter:
        if any(band + '_seeing_quality' in fl for fl in glob.glob(os.path.join(baseDir, "*"))):
            seeingFiles += sorted(
                glob.glob(
                    os.path.join(
                        baseDir,
                        '*'+band+'_seeing_quality*'
                    )
                )
            )
            referenceImages += sorted(
                glob.glob(
                    os.path.join(
                        baseDir,
                        '*'+band+'_key_images*'
                    )
                )
            )
            continue
        workingDirectory = glob.glob(
            os.path.join(
                baseDir,
                '**',
                'level0',
                '**',
                '*'+band+'*'
            ), recursive=True
        )
        if len(workingDirectory) == 0:
            continue
        workingDirectory = workingDirectory[0]
        # For the record, the length of this list comprehension is a bit.
        # It's stupid on purpose.
        # Don't judge me.
        obsSeries = [
            series for series in set(
                [
                    "_".join(z.split("_")[:-1]) for z in
                    [
                        q for q in glob.glob(
                            os.path.join(workingDirectory, '*fit*')
                        ) if 'dark' not in q and 'flat' not in q
                    ]
                ]
            ) if len(glob.glob(series + "*fit*")) > 10
        ]
        # For the record, though, it finds the series of ROSA files in workingDirectory
        # That aren't darks/flats with more than 10 files in the series.

        for series in obsSeries:
            flist = sorted(glob.glob(series + "*fit*"))
            seeingParams = []
            for file in flist:
                params = _rosa_params(file)
                seeingParams.append(params)
            print("Computing ROSA ", band, "Seeing Params")
            with ProgressBar():
                results = dask.compute(seeingParams)[0]

            exts = [x[0] for x in results]
            timestamps = np.concatenate([x[1] for x in results])
            hsm = np.concatenate([x[2] for x in results])
            hsmCenter = np.concatenate([x[3] for x in results]).flatten()
            centralPixel = np.concatenate([x[4] for x in results]).flatten()
            centralSquare = np.concatenate([x[5] for x in results])
            frameMean = np.concatenate([y for y in [x[6] for x in results]])
            frameRMS = np.concatenate([y for y in [x[7] for x in results]])
            centralRMS = np.concatenate([y for y in [x[8] for x in results]])

            tsCut = timestamps != 'nan'

            timestamps = timestamps[tsCut].astype('datetime64')
            tSort = np.argsort(timestamps)
            timestamps = timestamps[tSort]
            hsm = hsm[tsCut][tSort]
            hsmCenter = hsmCenter[tsCut][tSort]
            centralPixel = centralPixel[tsCut][tSort]
            centralSquare = centralSquare[tsCut][tSort]
            frameMean = frameMean[tsCut][tSort]
            frameRMS = frameRMS[tsCut][tSort]
            centralRMS = centralRMS[tsCut][tSort]

            # Separating our the Median and +/- 1-sigma images
            medianIndex = _find_nearest(np.nan_to_num(hsm), np.nanmedian(hsm))
            ctr = 0
            imgIndex = 0
            while ctr < medianIndex:
                ctr += exts[imgIndex]
                imgIndex += 1
            medianImage = fits.open(
                flist[imgIndex - 1]
            )[medianIndex - (ctr - exts[imgIndex - 1])].data

            minIndex = _find_nearest(
                np.nan_to_num(hsm),
                np.nanmean(hsm) - np.nanstd(hsm)
            )
            ctr = 0
            imgIndex = 0
            while ctr < minIndex:
                ctr += exts[imgIndex]
                imgIndex += 1
            minImage = fits.open(
                flist[imgIndex - 1]
            )[minIndex - (ctr - exts[imgIndex - 1])].data

            maxIndex = _find_nearest(
                np.nan_to_num(hsm),
                np.nanmean(hsm) + np.nanstd(hsm)
            )
            ctr = 0
            imgIndex = 0
            while ctr < maxIndex:
                ctr += exts[imgIndex]
                imgIndex += 1
            maxImage = fits.open(
                flist[imgIndex - 1]
            )[maxIndex - (ctr - exts[imgIndex - 1])].data

            seeingArray = np.rec.fromarrays(
                [
                    timestamps,
                    hsm,
                    hsmCenter,
                    centralPixel,
                    centralSquare,
                    frameMean,
                    frameRMS,
                    centralRMS
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

            obsDate = series.split("_")[-2].replace("-", "")
            obsTime = series.split("_")[-1].replace(":", "")

            seeingSaveStr = os.path.join(baseDir, obsDate + "_" + obsTime + "_" + band + "_seeing_quality.npy")
            refimSaveStr = os.path.join(baseDir, obsDate + "_" + obsTime + "_" + band + "_key_images.npy")

            seeingFiles.append(seeingSaveStr)
            referenceImages.append(refimSaveStr)

            np.save(seeingSaveStr, seeingArray)
            np.save(refimSaveStr, keyImages)
    return seeingFiles, referenceImages


def ibis_seeing_quality(baseDir):
    """
    Searches for IBIS seeing parameter and reference image files.
    If it can find them, returns the list of each type.
    Otherwise, searches for IBIS whitelight data and files,
    then determines seeing parameters for them, creates those files, and saves them.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD

    Returns:
    --------
    seeingFiles : list
        List of all IBIS seeing parameter files found
    referenceImages : list
        List of files containing IBIS reference images
    """

    seeingFiles = []
    referenceImages = []
    if any('ibis_seeing_quality' in fl for fl in glob.glob(os.path.join(baseDir, "*"))):
        seeingFiles += sorted(
            glob.glob(
                os.path.join(
                    baseDir,
                    '*ibis_seeing_quality*'
                )
            )
        )
        referenceImages += sorted(
            glob.glob(
                os.path.join(
                    baseDir,
                    '*ibis_key_images*'
                )
            )
        )
        return seeingFiles, referenceImages
    whitelightDirs = glob.glob(
        os.path.join(
            baseDir,
            'level0',
            "**",
            "*whitelight",
            "ScienceObservation",
            "*"
        )
    )
    for series in whitelightDirs:
        flist = sorted(glob.glob(os.path.join(series, "s*.fits")))
        datetimeString = series.split("/")[-2]
        seeingParams = []
        for file in flist:
            params = _ibis_params(file)
            seeingParams.append(params)
        print("Computing IBIS Seeing Params")
        with ProgressBar():
            results = dask.compute(seeingParams)[0]

        exts = [x[0] for x in results]
        timestamps = np.concatenate([x[1] for x in results])
        hsm = np.concatenate([x[2] for x in results])
        centralPixel = np.concatenate([x[3] for x in results]).flatten()
        centralSquare = np.concatenate([x[4] for x in results])
        frameMean = np.concatenate([y for y in [x[5] for x in results]])
        frameRMS = np.concatenate([y for y in [x[6] for x in results]])

        tsCut = timestamps != 'nan'

        timestamps = timestamps[tsCut].astype('datetime64')
        tSort = np.argsort(timestamps)
        timestamps = timestamps[tSort]
        hsm = hsm[tsCut][tSort]
        centralPixel = centralPixel[tsCut][tSort]
        centralSquare = centralSquare[tsCut][tSort]
        frameMean = frameMean[tsCut][tSort]
        frameRMS = frameRMS[tsCut][tSort]

        # Separating our the Median and +/- 1-sigma images
        medianIndex = _find_nearest(np.nan_to_num(hsm), np.nanmedian(hsm))
        ctr = 0
        imgIndex = 0
        while ctr < medianIndex:
            ctr += exts[imgIndex]
            imgIndex += 1
        medianImage = fits.open(
            flist[imgIndex - 1]
        )[medianIndex - (ctr - exts[imgIndex - 1])].data

        minIndex = _find_nearest(
            np.nan_to_num(hsm),
            np.nanmean(hsm) - np.nanstd(hsm)
        )
        ctr = 0
        imgIndex = 0
        while ctr < minIndex:
            ctr += exts[imgIndex]
            imgIndex += 1
        minImage = fits.open(
            flist[imgIndex - 1]
        )[minIndex - (ctr - exts[imgIndex - 1])].data

        maxIndex = _find_nearest(
            np.nan_to_num(hsm),
            np.nanmean(hsm) + np.nanstd(hsm)
        )
        ctr = 0
        imgIndex = 0
        while ctr < maxIndex:
            ctr += exts[imgIndex]
            imgIndex += 1
        maxImage = fits.open(
            flist[imgIndex - 1]
        )[maxIndex - (ctr - exts[imgIndex - 1])].data

        seeingArray = np.rec.fromarrays(
            [
                timestamps,
                hsm,
                centralPixel,
                centralSquare,
                frameMean,
                frameRMS,
            ],
            names=[
                'TIMESTAMPS',
                'HSM',
                'CENTRAL_PIXEL',
                'CENTRAL_MEAN',
                'IMAGE_MEAN',
                'IMAGE_RMS',
            ]
        )

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
        seeingSaveStr = os.path.join(baseDir, datetimeString + "_ibis_seeing_quality.npy")
        refimSaveStr = os.path.join(baseDir, datetimeString + "_ibis_key_images.npy")

        seeingFiles.append(seeingSaveStr)
        referenceImages.append(refimSaveStr)

        np.save(seeingSaveStr, seeingArray)
        np.save(refimSaveStr, keyImages)
    return seeingFiles, referenceImages


def zyla_seeing_quality(baseDir, obsDate, zylaShape=None):
    """
    Searches for Zyla/HARDCAM seeing parameter and reference image files.
    If it can find them, returns the list of each type.
    Otherwise, searches for data and files,
    then determines seeing parameters for them, creates those files, and saves them.

    Parameters:
    -----------
    baseDir : str
        Base observation directory. Pattern is typical /sunspot/solardata/YYYY/MM/DD
    obsDate : str
        Observation date in format YYYY-MM-DD
    zylaShape : None or tuple
        Keyword argument passthrough for the MFGS determination.
        Required if Zyla was operated in windowed mode

    Returns:
    --------
    seeingFiles : list
        List of all IBIS seeing parameter files found
    referenceImages : list
        List of files containing IBIS reference images
    """
    seeingFiles = []
    referenceImages = []
    if any('zyla_seeing_quality' in fl for fl in glob.glob(os.path.join(baseDir, '*'))):
        seeingFiles += sorted(
            glob.glob(
                os.path.join(
                    baseDir,
                    '*zyla_seeing_quality*'
                )
            )
        )
        referenceImages += sorted(
            glob.glob(
                os.path.join(
                    baseDir,
                    '*zyla_key_images*'
                )
            )
        )
        return seeingFiles, referenceImages
    # TEMP, REMOVE LATER
    else:
        return [], []

    zylaObssum = obssum.create_zyla_obssum(baseDir, obsDate)
    for i in range(len(zylaObssum['type'])):
        if 'map' not in zylaObssum['type'][i]:
            continue
        workingDirectory = zylaObssum['directory'][i]
        flist = zt.order_zyla_filelist(
            glob.glob(
                os.path.join(workingDirectory, "*.dat*")
            )
        )
        seeingParams = []
        for file in flist:
            params = _zyla_params(file)
            seeingParams.append(params)
        print("Computing Zyla Seeing Params")
        with ProgressBar():
            results = dask.compute(seeingParams)[0]
        np.save("tmp.npy", results)
        fnames = [x[0] for x in results]
        fname_sort = zt.argsort_zyla_filelist(fnames)
        fnames = np.array(fnames)[fname_sort]
        mfgs = np.array([x[1] for x in results])[fname_sort]
        centralPixel = np.array([x[2] for x in results])[fname_sort]
        centralSquare = np.array([x[3] for x in results])[fname_sort]
        frameMean = np.array([x[4] for x in results])[fname_sort]
        frameRMS = np.array([x[5] for x in results])[fname_sort]
        centralRMS = np.array([x[6] for x in results])[fname_sort]

        if np.isnan(zylaObssum['start'][i]):
            timetag = workingDirectory.split("/")[-2]
        else:
            timetag = zylaObssum['start'][i].astype(
                'datetime64[s]'
            ).astype(str).replace("-", "").replace("T", "_").replace(":", "")

        medianIndex = _find_nearest(np.nan_to_num(mfgs), np.nanmedian(mfgs))
        minIndex = _find_nearest(np.nan_to_num(mfgs), np.nanmean(mfgs) - np.nanstd(mfgs))
        maxIndex = _find_nearest(np.nan_to_num(mfgs), np.nanmean(mfgs) + np.nanstd(mfgs))
        if 'fits' in fnames[medianIndex]:
            medianImage = fits.open(fnames[medianIndex])[1].data
            minImage = fits.open(fnames[minIndex])[1].data
            maxImage = fits.open(fnames[maxIndex])[1].data
        else:
            medianImage = zt.read_zyla(
                fnames[medianIndex],
                dataShape=zylaShape,
                imageShape=zylaShape
            )
            minImage = zt.read_zyla(
                fnames[minIndex],
                dataShape=zylaShape,
                imageShape=zylaShape
            )
            maxImage = zt.read_zyla(
                fnames[maxIndex],
                dataShape=zylaShape,
                imageShape=zylaShape
            )

        seeingArray = np.rec.fromarrays(
            [
                fnames,
                mfgs,
                centralPixel,
                centralSquare,
                frameMean,
                frameRMS,
                centralRMS
            ],
            names=[
                'FILENAME',
                'MFGS',
                'CENTRAL_PIXEL',
                'CENTRAL_MEAN',
                'IMAGE_MEAN',
                'IMAGE_RMS',
                'CENTRAL_RMS'
            ]
        )

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
        seeingSaveStr = os.path.join(baseDir, timetag + "_zyla_seeing_quality.npy")
        refimSaveStr = os.path.join(baseDir, timetag + "_zyla_key_images.npy")

        seeingFiles.append(seeingSaveStr)
        referenceImages.append(refimSaveStr)

        np.save(seeingSaveStr, seeingArray)
        np.save(refimSaveStr, keyImages)
    return seeingFiles, referenceImages
