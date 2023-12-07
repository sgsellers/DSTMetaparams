import numpy as np
import os
import re


def rosa_zyla_detect_zyla_dims(imageData):
    """
    Detects the data and image dimensions in Zyla unformatted
    binary image files.

    Parameters
    ----------
    imageData : numpy.ndarray
        A one-dimensional Numpy array containing image data.
    """

    # Detects data then usable image dimensions.
    # Assumes all overscan regions within the raw
    # image has a zero value. If no overscan,
    # this function will fail. Will also fail if
    # dead pixels are present.
    def rosa_zyla_detect_overscan():
        # Borrowed from a Stackoverflow article
        # titled "Finding the consecutive zeros
        # in a numpy array." Author unknown.
        # Create an array that is 1 where imageData is 0,
        # and pad each end with an extra 0.
        # LATER: check for identical indices which would
        # be indicative of non-contiguous dead
        # pixels.
        iszero = np.concatenate(([0],
                                 np.equal(imageData, 0).view(np.int8),
                                 [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0]
        return ranges

    # Detects image columns by looking for the
    # last overscan pixel in the first row. Rows
    # are computed from the quotient of the
    # number of pixels and the number of columns.
    # Get data dimensions.
    # LATER: add function to check for overscan.
    #   Could be done by examining the number
    # 	of results in ovrScn.
    # self.logger.info("Attempting to detect overscan and data shape.")
    ovrScn = rosa_zyla_detect_overscan()
    datDim = (np.uint16(imageData.size / ovrScn[1]),
              ovrScn[1])
    # Detects usable image columns by using
    # the first overscan index. Finds first
    # overscan row by looking for the first
    # occurance where overscan indices are
    # not separated by dx1 or dx2.
    # Get image dimensions.
    # self.logger.info("Attempting to detect image shape.")
    dx1 = ovrScn[1] - ovrScn[0]
    dx2 = ovrScn[2] - ovrScn[1]
    DeltaX = np.abs(np.diff(ovrScn))
    endRow = (np.where(
        np.logical_and(
            DeltaX != dx1,
            DeltaX != dx2
        )
    )[0])[0] / 2 + 1
    imgDim = (np.uint16(endRow), ovrScn[0])

    return datDim, imgDim


def read_zyla(file, dataShape=None, imageShape=None, dtype=np.uint16, expand=True):
    """
    Reads an unformatted binary file. Slices the image as
    s[i] ~ 0:imageShape[i].

    Parameters
    ----------
    file : str
        Path to binary image file.
    dataShape : tuple
        Shape of the image or cube.
    imageShape : tuple
        Shape of sub-image or region of interest.
    dtype : Numpy numerical data type.
        Default is numpy.uint16.

    Returns
    -------
    numpy.ndarray : np.float32, shape imageShape.
    """
    try:
        with open(file, mode='rb') as imageFile:
            imageData = np.fromfile(imageFile,
                                    dtype=dtype
                                    )
    except Exception as err:
        print("Could not open/read binary image file: "
              "{0}".format(err)
              )
        raise

    if dataShape is None:
        dataShape, _ = rosa_zyla_detect_zyla_dims(imageData)

    if imageShape is None:
        _, imageShape = rosa_zyla_detect_zyla_dims(imageData)

    im = imageData.reshape(dataShape)
    # Generate a tuple of slice objects, s[i]~ 0:imageShape[i]
    s = tuple()
    for t in imageShape:
        s = s + np.index_exp[0:t]
    im = im[s]
    if expand:
        return np.float32(im)
    else:
        return im


def order_zyla_filelist(flist):
    """Zyla files order by the least significant digit. This takes a filelist and sorts it to be sequential"""

    orderlist = [''] * len(flist)
    pattern = '[0-9]+'
    p = re.compile(pattern)
    for f in flist:
        head, tail = os.path.split(f)
        match = p.match(tail)
        digitnew = match.group()[::-1]
        orderlist[int(digitnew)] = f
    return orderlist


def argsort_zyla_filelist(flist):
    """Returns list that, applied as indices to numpy array, will apply zyla filename sorting"""
    orderlist = [''] * len(flist)
    pattern = '[0-9]+'
    p = re.compile(pattern)
    for f in range(len(flist)):
        head, tail = os.path.split(flist[f])
        match = p.match(tail)
        digitnew = match.group()[::-1]
        orderlist[int(digitnew)] = f
    return orderlist
