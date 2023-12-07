"""
Series of functions for determining what the telescope is pointing at.
The currently provided functions will determine if there are any of the following in the maximum
FOV. The maximum FOV is 175" approximately.
    1.) Active regions (HEK)
        a.) Will include NOAA number if there is one available
    2.) Coronal Hole (HEK)
    3.) Sigmoids (HEK)
    3.) Filament (HEK) EDIT: SCRATCH THAT. HEK coverage of filaments is *poor*. 
        We'll just have to assume that filament datasets contain filaments, and everything else may not.    
    4.) Flare of B1.0 class or greater (HEK)
    5.) Filament Eruption (HEK)
    6.) Coronal Rain (HEK)
    7.) Solar limb (Simple Coordinate Comparison)
    8.) IRIS co-observations (VSO)
Most of these rely in some way on sunpy's net functions.
Filaments, eruptions, rain, sigmoids, and flares are handled by HEK as point sources.
Coronal Holes and Active Regions are treated as extended sources.
"""

import numpy as np
import matplotlib.path as mpath
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.net import Fido, hek, attrs as a
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.coordinates import frames

# Full FOV for Port 4 Gray Box field stop. Change this if we ever rip that stop out.
defaultFov = (175, 175)  # arcseconds

# Since it's non-trivial to pull IRIS pointings, this is a fudge.
irisRadius = 120  # arcseconds


def HEKRegionIntersection(obsTime, sLon, sLat, fov=defaultFov, minCHArea=5e10, minARArea=1e10):
    """Checks HEK database for all Coronal Holes and Active Regions on the date of obsTime,
    Rotates their coordinates from their detection time, and checks for intersections between
    the rotated coordinates, and the telescope FOV, as determined by SLON, SLAT, obsTime, and FOV.
    If there are any intersections, the type of region, and, if an AR with a NOAA number,
    the NOAA number of the intersecting region(s) will be returned.
    If the intersecting region is an AR with no NOAA number, it will be tagged as "Unnamed AR".

    Parameters:
    -----------
    obsTime : str
        Start time of observing series
    sLon : float
        Stonyhurst Longitude of observing series.
    sLat : float
        Stonyhurst Latitude of observing series
    fov : tuple
        Tuple of telescope FOV (Fov(X),Fov(Y)) in arcsec. Default is (175,175)
    minCHArea : float
        Minimum area at disk center to count as a CH. Avoids spurious detections
    minARArea : float
        Minimum area at disk center to count as an AR for ARS NOT tagged with a NOAA num.
    
    Returns:
    --------
    intersectingRegions : dictionary
        Dictionary of intersection region types, identifiers, XY Center Coordinates
    """

    obsDate = np.datetime64(obsTime, "D").astype(str)
    try:
        hekARSearch = hek.HEKClient().search(
                a.Time(obsDate + " 00:00", obsDate + " 23:59"),
                a.hek.AR
        )
    except (Exception,):
        return {"AR Nums": [],
                "AR Centers": [],
                "CH Centers": []}
    hekCHSearch = hek.HEKClient().search(
            a.Time(obsDate + " 00:00", obsDate + " 23:59"),
            a.hek.CH,
            a.hek.FRM.Name == 'SPoCA'
    )

    telescopePointing = SkyCoord(
            sLon * u.deg,
            sLat * u.deg,
            frame=frames.HeliographicStonyhurst,
            observer='earth',
            obstime=obsTime
    ).transform_to(frames.Helioprojective)

    xLeft = float((telescopePointing.Tx - (fov[0]/2)*u.arcsec).value)
    xRight = float((telescopePointing.Tx + (fov[0]/2)*u.arcsec).value)
    yBottom = float((telescopePointing.Ty - (fov[1]/2)*u.arcsec).value)
    yTop = float((telescopePointing.Ty + (fov[1]/2)*u.arcsec).value)

    fovPath = mpath.Path(
            np.array(
                [
                    (xLeft, yBottom),
                    (xRight, yBottom),
                    (xRight, yTop),
                    (xLeft, yTop)
                ]
            )
    )

    activeRegionCenters = []
    activeRegionNumbers = []
    for response in hekARSearch:
        # For some reason named ARs don't have boundcc.
        if response['hpc_bbox'] == '':
            continue
        ARBounds = [
                (
                    float(v.split(" ")[0]), 
                    float(v.split(" ")[1])
                ) for v in response['hpc_bbox'][9:-2].split(",")
        ]
        ARSkycoord = SkyCoord(
                [pair * u.arcsec for pair in ARBounds],
                obstime=response['event_starttime'].value,
                observer='earth',
                frame=frames.Helioprojective
        )
        rotatedARCoord = solar_rotate_coordinate(ARSkycoord, time=obsTime)
        ARPath = mpath.Path(
                np.array(
                    [(i.Tx.value, i.Ty.value) for i in rotatedARCoord]
                )
        )
        if ARPath.intersects_path(fovPath):
            if not response['ar_noaanum']:
                if not response['area_atdiskcenter']:
                    continue
                if response['area_atdiskcenter'] >= minARArea:
                    activeRegionNumbers.append("Unnamed AR")
                    activeRegionCenters.append(tuple(np.nanmean(ARPath.vertices, axis=0)))
            else:
                activeRegionNumbers.append(response['ar_noaanum'])
                activeRegionCenters.append(tuple(np.nanmean(ARPath.vertices, axis=0)))
    coronalHoleCenter = []
    for response in hekCHSearch:
        if response['hpc_boundcc'] == '':
            continue
        CHBounds = [
                (
                    float(v.split(" ")[0]),
                    float(v.split(" ")[1])
                ) for v in response['hpc_boundcc'][9:-2].split(",")
        ]
        CHSkycoord = SkyCoord(
                [pair * u.arcsec for pair in CHBounds],
                obstime=response['event_starttime'].value,
                observer='earth',
                frame=frames.Helioprojective
        )
        rotatedCHCoord = solar_rotate_coordinate(CHSkycoord, time=obsTime)
        CHPath = mpath.Path(
                np.array(
                    [(i.Tx.value, i.Ty.value) for i in rotatedCHCoord]
                )
        )
        # Fudge factor to only select decently-sized CH
        area = response['area_atdiskcenter']
        if (CHPath.intersects_path(fovPath)) & (area >= minCHArea):
            coronalHoleCenter.append(tuple(np.nanmean(CHPath.vertices, axis=0)))
    intersectingRegions = {"AR Nums": activeRegionNumbers,
                           "AR Centers": activeRegionCenters,
                           "CH Centers": coronalHoleCenter}
    return intersectingRegions


def HEKEventIntersection(obsTime, endTime, sLon, sLat, fov=defaultFov):
    """Checks HEK database for all: 
        Flares, Filament Eruptions, Coronal Rains and Sigmoids 
    On the date of obsTime within a 30 minute interval surrounding the interval,
    Rotates their coordinates from their detection time, and checks for intersections between
    the rotated coordinates, and the telescope FOV, as determined by SLON, SLAT, obsTime, and FOV.
    If there are any intersections, the type of event and its coordinates will be returned,
    as well as its quality factor.
    If a flare, its class is returned as well.
    The quality parameter is, again, 
        2 for an event in FOV and time.
        1 for an event in either FOV or time
        0.5 for an event near FOV and time.
    Given that I'm not sure if I care about sigmoids, I'm going to make an executive decision here.
    Sigmoids are capped at a 1-bump for quality.
    FUTURE SEAN/OTHER PERSON: I checked the frequency of the other events in HEK.
    They're not worth including. If you know a better, query-able database for them, please integrate.

    Parameters:
    -----------
    obsTime : str
        Start time of observing series
    endTime : str
        End time of observing series
    sLon : float
        Stonyhurst Longitude of observing series.
    sLat : float
        Stonyhurst Latitude of observing series
    fov : tuple
        Tuple of telescope FOV (Fov(X),Fov(Y)) in arcsec. Default is (175,175)
    
    Returns:
    --------
    intersectingRegions : dictionary
        Dictionary of intersection region types, identifiers, XY Center Coordinates
    """

    searchStart = np.datetime64(obsTime) - np.timedelta64(30, "m")
    searchEnd = np.datetime64(endTime) + np.timedelta64(30, "m")
    try:
        telescopePointing = SkyCoord(
                sLon * u.deg,
                sLat * u.deg,
                frame=frames.HeliographicStonyhurst,
                observer='earth',
                obstime=obsTime
        ).transform_to(frames.Helioprojective)
    except (Exception,):
        return {
            "Flare Type": [],
            "Flare Center": [],
            "Flare Quality": [],
            "Filament Eruption Center": [],
            "Filament Eruption Quality": [],
            "Coronal Rain Center": [],
            "Coronal Rain Quality": [],
            "Sigmoid Center": []
        }

    xLeft = float((telescopePointing.Tx - (fov[0]/2)*u.arcsec).value)
    xRight = float((telescopePointing.Tx + (fov[0]/2)*u.arcsec).value)
    yBottom = float((telescopePointing.Ty - (fov[1]/2)*u.arcsec).value)
    yTop = float((telescopePointing.Ty + (fov[1]/2)*u.arcsec).value)

    fovPath = mpath.Path(
            np.array([(xLeft, yBottom), (xRight, yBottom), (xRight, yTop), (xLeft, yTop)])
    )

    fovExpandedPath = mpath.Path.circle(
            center=(telescopePointing.Tx.value, telescopePointing.Ty.value),
            radius=100
    )

    hekFLSearch = hek.HEKClient().search(
            a.Time(searchStart.astype(str), searchEnd.astype(str)),
            a.hek.FL,
            a.hek.FL.GOESCls > 'B1.0'
    )
    hekFESearch = hek.HEKClient().search(
            a.Time(searchStart.astype(str), searchEnd.astype(str)),
            a.hek.FE
    )
    hekCRSearch = hek.HEKClient().search(
            a.Time(searchStart.astype(str), searchEnd.astype(str)),
            a.hek.CR
    )
    hekSGSearch = hek.HEKClient().search(
            a.Time(searchStart.astype(str), searchEnd.astype(str)),
            a.hek.SG
    )

    flareClass = []
    flareCenters = []
    flareQuality = []
    for response in hekFLSearch:
        # Checking if it's a valid event...
        # Case: No coords, skip loop iteration
        if response['event_coord1'] == '' or response['event_coord2'] == '':
            continue
        # Case: No peaktime, but coords. Default to startObs
        elif response['event_peaktime'] == '':
            flarePeaktime = obsTime
        # Case: Peaktime and coords, all is right and proper.
        else:
            flarePeaktime = str(response['event_peaktime'].value)

        if response['event_coordunit'] == 'arcsec':
            unit = u.arcsec
            frm = frames.Helioprojective
        else:
            # Default to degrees as unit of measurement.
            unit = u.deg
            frm = frames.HeliographicStonyhurst
        flareCoord = SkyCoord(
                response['event_coord1']*unit,
                response['event_coord2']*unit,
                obstime=flarePeaktime,
                observer='earth',
                frame=frm
        ).transform_to(frames.Helioprojective)
        rotated_flare_coord = solar_rotate_coordinate(flareCoord, time=obsTime)
        flarePath = mpath.Path(
            np.array([(rotated_flare_coord.Tx.value, rotated_flare_coord.Ty.value)]).reshape(1, 2)
        )
        flarePeaktime = np.datetime64(flarePeaktime)
        if ((flarePath.intersects_path(fovPath)) &
                (flarePeaktime >= np.datetime64(obsTime)) &
                (flarePeaktime <= np.datetime64(endTime))):
            # Best case scenario: in FOV and peak in obs interval
            flareClass.append(response['fl_goescls'])
            flareCenters.append(
                (rotated_flare_coord.Tx.value, rotated_flare_coord.Ty.value)
            )
            flareQuality.append(2)
        elif flarePath.intersects_path(fovPath):
            # In field, out of time, second best case.
            flareClass.append(response['fl_goescls'])
            flareCenters.append(
                (rotated_flare_coord.Tx.value, rotated_flare_coord.Ty.value)
            )
            flareQuality.append(1)
        elif ((flarePath.intersects_path(fovExpandedPath)) &
              (flarePeaktime >= np.datetime64(obsTime)) &
              (flarePeaktime <= np.datetime64(endTime))):
            # Out of field, nearby, in time
            flareClass.append(response['fl_goescls'])
            flareCenters.append(
                (rotated_flare_coord.Tx.value, rotated_flare_coord.Ty.value)
            )
            flareQuality.append(1)
        elif flarePath.intersects_path(fovExpandedPath):
            # By selection criteria, everything must be within 30 minutes.
            # This is the worst scenario -- it's nearby but out of frame in space/time
            flareClass.append(response['fl_goescls'])
            flareCenters.append(
                (rotated_flare_coord.Tx.value, rotated_flare_coord.Ty.value)
            )
            flareQuality.append(0.5)

    # Filament Eruptions
    filamentEruptionCenters = []
    filamentEruptionQuality = []
    for response in hekFESearch:
        # Checking if it's a valid event...
        # Case: No coords, skip loop iteration
        if response['event_coord1'] == '' or response['event_coord2'] == '':
            continue
        # Case: No peaktime, but coords. Default to startObs
        elif response['event_peaktime'] == '':
            fePeaktime = obsTime
        # Case: Peaktime and coords, all is right and proper.
        else:
            fePeaktime = str(response['event_peaktime'].value)

        if response['event_coordunit'] == 'arcsec':
            unit = u.arcsec
            frm = frames.Helioprojective
        else:
            # Default to degrees as unit of measurement.
            unit = u.deg
            frm = frames.HeliographicStonyhurst
        feCoord = SkyCoord(
            response['event_coord1'] * unit,
            response['event_coord2'] * unit,
            obstime=fePeaktime,
            observer='earth',
            frame=frm
        ).transform_to(frames.Helioprojective)
        rotated_fe_coord = solar_rotate_coordinate(feCoord, time=obsTime)
        fePath = mpath.Path(
            np.array([(rotated_fe_coord.Tx.value, rotated_fe_coord.Ty.value)]).reshape(1, 2)
        )
        fePeaktime = np.datetime64(fePeaktime)
        if ((fePath.intersects_path(fovPath)) &
                (fePeaktime >= np.datetime64(obsTime)) &
                (fePeaktime <= np.datetime64(endTime))):
            # Best case scenario: in FOV and peak in obs interval
            filamentEruptionCenters.append(
                (rotated_fe_coord.Tx.value, rotated_fe_coord.Ty.value)
            )
            filamentEruptionQuality.append(2)
        elif fePath.intersects_path(fovPath):
            # In field, out of time, second best case.
            filamentEruptionCenters.append(
                (rotated_fe_coord.Tx.value, rotated_fe_coord.Ty.value)
            )
            filamentEruptionQuality.append(1)
        elif ((fePath.intersects_path(fovExpandedPath)) &
              (fePeaktime >= np.datetime64(obsTime)) &
              (fePeaktime <= np.datetime64(endTime))):
            # Out of field, nearby, in time
            filamentEruptionCenters.append(
                (rotated_fe_coord.Tx.value, rotated_fe_coord.Ty.value)
            )
            filamentEruptionQuality.append(1)
        elif fePath.intersects_path(fovExpandedPath):
            # By selection criteria, everything must be within 30 minutes.
            # This is the worst scenario -- it's nearby but out of frame in space/time
            filamentEruptionCenters.append(
                (rotated_fe_coord.Tx.value, rotated_fe_coord.Ty.value)
            )
            filamentEruptionQuality.append(0.5)

    # Coronal Rain
    coronalRainCenters = []
    coronalRainQuality = []
    for response in hekCRSearch:
        # Checking if it's a valid event...
        # Case: No coords, skip loop iteration
        if response['event_coord1'] == '' or response['event_coord2'] == '':
            continue

        if response['event_coordunit'] == 'arcsec':
            unit = u.arcsec
            frm = frames.Helioprojective
        else:
            # Default to degrees as unit of measurement.
            unit = u.deg
            frm = frames.HeliographicStonyhurst
        crCoord = SkyCoord(
            response['event_coord1'] * unit,
            response['event_coord2'] * unit,
            obstime=obsTime,  # Coronal Rain events aren't always tagged with a time.
            observer='earth',
            frame=frm
        ).transform_to(frames.Helioprojective)
        crPath = mpath.Path(
            np.array([(crCoord.Tx.value, crCoord.Ty.value)]).reshape(1, 2)
        )
        if crPath.intersects_path(fovPath):
            # Best case scenario: in FOV and peak in obs interval
            coronalRainCenters.append(
                (crCoord.Tx.value, crCoord.Ty.value)
            )
            coronalRainQuality.append(2)
        elif crPath.intersects_path(fovExpandedPath):
            # Out of field, nearby, in time
            coronalRainCenters.append(
                (crCoord.Tx.value, crCoord.Ty.value)
            )
            coronalRainQuality.append(1)

    # Sigmoids
    sigmoidCenters = []
    for response in hekSGSearch:
        # Checking if it's a valid event...
        # Case: No coords, skip loop iteration
        if response['event_coord1'] == '' or response['event_coord2'] == '':
            continue

        if response['event_coordunit'] == 'arcsec':
            unit = u.arcsec
            frm = frames.Helioprojective
        else:
            # Default to degrees as unit of measurement.
            unit = u.deg
            frm = frames.HeliographicStonyhurst
        sgCoord = SkyCoord(
            response['event_coord1'] * unit,
            response['event_coord2'] * unit,
            obstime=obsTime,  # Sigmoids aren't always tagged with a time.
            observer='earth',
            frame=frm
        ).transform_to(frames.Helioprojective)
        sgPath = mpath.Path(
            np.array([(sgCoord.Tx.value, sgCoord.Ty.value)]).reshape(1, 2)
        )
        if sgPath.intersects_path(fovExpandedPath):
            # Since we're not tagging this for quality, any nearby can be entered
            sigmoidCenters.append(
                (sgCoord.Tx.value, sgCoord.Ty.value)
            )

    intersectingEvents = {
        "Flare Type": flareClass,
        "Flare Center": flareCenters,
        "Flare Quality": flareQuality,
        "Filament Eruption Center": filamentEruptionCenters,
        "Filament Eruption Quality": filamentEruptionQuality,
        "Coronal Rain Center": coronalRainCenters,
        "Coronal Rain Quality": coronalRainQuality,
        "Sigmoid Center": sigmoidCenters
    }

    return intersectingEvents


def IRISIntersection(obsTime, endTime, sLon, sLat, DSTfov=defaultFov, IRISfov=irisRadius):
    """Checks VSO for IRIS observations.
    Since finding the actual FOV observed by IRIS is nontrivial (it's nontrivial for the Dunn as well),
    We naively check if a circle of the defined radius intersects the DSTfov square box.
    Returns the number of intersecting IRIS obs

    Parameters:
    -----------
    obsTime : str
        DST observation start
    endTime : str
        DST observation end
    """

    obsDate = np.datetime64(obsTime, "D").astype(str)
    obsTimeDT = np.datetime64(obsTime, 's').astype(int)
    endTimeDT = np.datetime64(endTime, 's').astype(int)
    try:
        irisSearch = Fido.search(a.Time(obsDate + " 00:00", obsDate + " 23:59"), a.Instrument("IRIS"))[0]
        telescopePointing = SkyCoord(
            sLon * u.deg,
            sLat * u.deg,
            frame=frames.HeliographicStonyhurst,
            observer='earth',
            obstime=obsTime
        ).transform_to(frames.Helioprojective)
    except (Exception,):
        return 0

    xLeft = float((telescopePointing.Tx - (DSTfov[0] / 2) * u.arcsec).value)
    xRight = float((telescopePointing.Tx + (DSTfov[0] / 2) * u.arcsec).value)
    yBottom = float((telescopePointing.Ty - (DSTfov[1] / 2) * u.arcsec).value)
    yTop = float((telescopePointing.Ty + (DSTfov[1] / 2) * u.arcsec).value)

    fovPath = mpath.Path(
        np.array([(xLeft, yBottom), (xRight, yBottom), (xRight, yTop), (xLeft, yTop)])
    )

    # Number of intersecting unique obs
    irisIntersectingTime = []
    for response in irisSearch:
        irisCoord = SkyCoord(
            float(response['Extent X']) * u.arcsec,
            float(response['Extent Y']) * u.arcsec,
            obstime=response['Start Time'].value,
            observer='earth',
            frame=frames.Helioprojective
        )
        irisRotatedCoord = solar_rotate_coordinate(irisCoord, time=obsTime)
        irisPath = mpath.Path.circle(
            center=(irisRotatedCoord.Tx.value, irisRotatedCoord.Ty.value),
            radius=IRISfov/2
        )
        irisStart = np.datetime64(response['Start Time'].value, 's').astype(int)
        irisEnd = np.datetime64(response['End Time'].value, 's').astype(int)
        if (irisPath.intersects_path(fovPath) &
                ((obsTimeDT in range(irisStart, irisEnd)) or
                 (endTimeDT in range(irisStart, irisEnd)))):
            irisIntersectingTime.append(
                np.datetime64(response['Start Time'].value, "m")
            )

    numIRISObs = len(list(set(irisIntersectingTime)))
    return numIRISObs
