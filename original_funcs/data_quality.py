import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import glob
import tqdm

from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a
import astropy.units as u

def determine_quality_metaparams(indir, date):
    """Sets up information needed for our quality metrics.
    Outputs a structure with:
    OBSTYPE
    DURATION
    HSM Coefficient of Variation (GBAND)
    HSM Coefficient of Variation (4170)
    HSM Coefficient of Variation (CAK)
    MFGS Coefficient of Variation (ZYLA)
    (CV is stdev/mean)
    EVENT FACTOR
    EVENT FACTOR TYPE
    """
    #First, check if there's pointing info
    pinfo = os.path.join(os.path.join(indir, "context_ims"),"pointing_info.txt")
    if not os.path.isfile(pinfo):
        return None
    pointings = np.recfromtxt(pinfo, delimiter=',',names=True,encoding=None)
    pointings = np.atleast_1d(pointings)
    if len(pointings.TIME) == 0:
        return None
    # Cool. There's pointing info.
    # Now, check if there's seeing info.
    all_seeing = sorted(glob.glob(os.path.join(indir, "*seeing_quality.npy")))
    if len(all_seeing) == 0:
        return None
    # Cool. There's seeing info.
    
    # OBSTYPE
    obsreg = open("/var/www/html/observations_registry.csv","r")
    oreg_lines = obsreg.readlines()
    obsreg.close()
    all_datestrs = [l.split(",")[1] for l in oreg_lines]
    all_otypes = [l.split(",")[2] for l in oreg_lines]
    obstype = all_otypes[all_datestrs.index(date)]
    
    duration = np.timedelta64(0,"m")
    for i in range(len(pointings.TIME)):
        duration += np.datetime64(pointings.END[i]) - np.datetime64(pointings.TIME[i])
    
    zyla_cv = []
    gband_cv = []
    cak_cv = []
    cont_cv = []
    ibis_cv = []
    
    for i in range(len(all_seeing)):
        if "zyla" in all_seeing[i]:
            see = np.load(all_seeing[i])
            zyla_cv.append(np.nanstd(see['MFGS'])/np.nanmean(see['MFGS']))
        else:
            try:
                see = np.load(all_seeing[i])
            except:
                see = np.load(all_seeing[i],allow_pickle=True).flat[0]
            if 'gband' in all_seeing[i]:
                gband_cv.append(np.nanstd(see['HSM'])/np.nanmean(see['HSM']))
            elif '4170_' in all_seeing[i]:
                cont_cv.append(np.nanstd(see['HSM'])/np.nanmean(see['HSM']))
            elif 'cak_' in all_seeing[i]:
                cak_cv.append(np.nanstd(see['HSM'])/np.nanmean(see['HSM']))
            elif 'ibis' in all_seeing[i]:
                ibis_cv.append(np.nanstd(see['HSM'])/np.nanmean(see['HSM']))
                
    if len(zyla_cv) == 0:
        zyla_cv = np.nan
    else:
        zyla_cv = np.nanmean(np.array(zyla_cv))
    if len(gband_cv) == 0:
        gband_cv = np.nan
    else:
        gband_cv = np.nanmean(np.array(gband_cv))
    if len(cak_cv) == 0:
        cak_cv = np.nan
    else:
        cak_cv = np.nanmean(np.array(cak_cv))
    if len(cont_cv) == 0:
        cont_cv = np.nan
    else:
        cont_cv = np.nanmean(np.array(cont_cv))
    if len(ibis_cv) == 0:
        ibis_cv = np.nan
    else:
        ibis_cv = np.nanmean(np.array(cont_cv))
    # Cool. We've got duration and seeing. 
    # Now for the hard part.
    # Our fudge factor for events
    
    # Conceivably, there might be multiple flares/other events in frame.
    # In this case, we want the max fudge factor, so append to a list
    # NOTE: Coronal rain doesn't necessarily have an associated time.
    # For CR, if there's an event, it's close enough temporally by default.
    event_factors = []
    events = []
    see_cv = []
    for i in range(len(pointings.TIME)):
        tele_pointing = SkyCoord(
            pointings['SLON'][i]*u.deg,
            pointings['SLAT'][i]*u.deg,
            frame=frames.HeliographicStonyhurst,
            observer='earth',
            obstime=pointings['TIME'][i]
        )
        tele_pointing = tele_pointing.transform_to(frames.Helioprojective)
        tele_start = np.datetime64(pointings.TIME[i])
        tele_end = np.datetime64(pointings.END[i])

        #### Real quick, add in scmon values...
        see_cv.append(float(pointings.SSCIN[i])/float(pointings.MSCIN[i]))
        
        #### Flares first.
        #### ...Cause I care about them the most.
        flare_fudge = []
        flare_events = []
        flare_query = Fido.search(
            a.Time(
                (tele_start - np.timedelta64(30,'m')).astype(str),
                (tele_end + np.timedelta64(30,'m')).astype(str)
            ),
            a.hek.EventType('FL'),
            a.hek.FL.GOESCls>'B1.0'
        )
        if len(flare_query['hek']) > 0:
            flare_xcd = flare_query['hek']['event_coord1']
            flare_ycd = flare_query['hek']['event_coord2']
            flare_peak = flare_query['hek']['event_peaktime']
            flare_class = flare_query['hek']['fl_goescls']
            flare_coordsys = flare_query['hek']['event_coordunit']
        else:
            flare_xcd = []
            flare_ycd = []
            flare_peak = []
            flare_class = []
            flare_coordsys = []
        for j in range(len(flare_xcd)):
            if flare_coordsys[j] == 'degrees':
                unit=u.deg
                frm = frames.HeliographicStonyhurst
            elif flare_coordsys[j] == 'arcsec':
                unit=u.arcsec
                frm=frames.Helioprojective
            flare_coord = SkyCoord(
                flare_xcd[j]*unit,
                flare_ycd[j]*unit,
                obstime=flare_peak[j],
                observer='earth',
                frame=frm
            )
            flare_coord = flare_coord.transform_to(frames.Helioprojective)
            del_x = float((tele_pointing.Tx - flare_coord.Tx).value)
            del_y = float((tele_pointing.Ty - flare_coord.Ty).value)
            distance = np.sqrt(del_x**2 + del_y**2)
            # Check if it's nearby:
            if distance <= 200:
                event_time = np.datetime64(flare_peak[j].value[0])
                # Best Case. Perfectly in field and on time
                if (event_time < tele_end) & (event_time > tele_start) & (distance <= 100):
                    flare_fudge.append(2)
                    flare_events.append(flare_class[j])
                # Second best case. Slightly out of time, but perfectly in field.
                elif (distance <= 100):
                    flare_fudge.append(1)
                    flare_events.append(flare_class[j])
                # Second best case. Perfectly in time, slightly out of field.
                elif (event_time < tele_end) & (event_time > tele_start):
                    flare_fudge.append(1)
                    flare_events.append(flare_class[j])
                # We've already passively selected this. Just out of FOV, and not in time.
                else:
                    flare_fudge.append(0.5)
                    flare_events.append(flare_class[j])
        # Check our flares. If there are any, append them to the event list.
        if len(flare_fudge) > 0:
            event_factors.append(np.nanmax(flare_fudge))
            events.append(flare_events[flare_fudge.index(np.nanmax(flare_fudge))])
        
        #### Filament eruptions Second.
        #### ...Cause they're like flares.
        filament_fudge = []
        filament_query = Fido.search(
            a.Time(
                (tele_start - np.timedelta64(30,'m')).astype(str),
                (tele_end + np.timedelta64(30,'m')).astype(str)
            ),
            a.hek.EventType('FE')
        )
        if len(filament_query['hek']) > 0:
            filament_xcd = filament_query['hek']['event_coord1']
            filament_ycd = filament_query['hek']['event_coord2']
            filament_peak = filament_query['hek']['event_peaktime']
            filament_coordsys = filament_query['hek']['event_coordunit']
        else:
            filament_xcd = []
            filament_ycd = []
            filament_peak = []
            filament_coordsys = []
        for j in range(len(filament_xcd)):
            if filament_coordsys[j] == 'degrees':
                unit=u.deg
                frm = frames.HeliographicStonyhurst
            elif filament_coordsys[j] == 'arcsec':
                unit=u.arcsec
                frm=frames.Helioprojective
            filament_coord = SkyCoord(
                filament_xcd[j]*unit,
                filament_ycd[j]*unit,
                obstime=filament_peak[j],
                observer='earth',
                frame=frm
            )
            filament_coord = filament_coord.transform_to(frames.Helioprojective)
            del_x = float((tele_pointing.Tx - filament_coord.Tx).value)
            del_y = float((tele_pointing.Ty - filament_coord.Ty).value)
            distance = np.sqrt(del_x**2 + del_y**2)
            # Check if it's nearby:
            if distance <= 200:
                event_time = np.datetime64(filament_peak[j].value[0])
                # Best Case. Perfectly in field and on time
                if (event_time < tele_end) & (event_time > tele_start) & (distance <= 100):
                    filament_fudge.append(2)
                # Second best case. Slightly out of time, but perfectly in field.
                elif (distance <= 100):
                    filament_fudge.append(1)
                # Second best case. Perfectly in time, slightly out of field.
                elif (event_time < tele_end) & (event_time > tele_start):
                    filament_fudge.append(1)
                # We've already passively selected this. Just out of FOV, and not in time.
                else:
                    filament_fudge.append(0.5)
        # Check our filaments. If there are any, append them to the event list.
        if len(filament_fudge) > 0:
            event_factors.append(np.nanmax(filament_fudge))
            events.append('Filament Eruption')
            
    
        #### Coronal Rain last
        #### Cause it's calm.
        rain_fudge = []
        rain_query = Fido.search(
            a.Time(
                (tele_start - np.timedelta64(30,'m')).astype(str),
                (tele_end + np.timedelta64(30,'m')).astype(str)
            ),
            a.hek.EventType('CR')
        )
        if len(rain_query['hek']) > 0:
            rain_xcd = rain_query['hek']['event_coord1']
            rain_ycd = rain_query['hek']['event_coord2']
            rain_coordsys = rain_query['hek']['event_coordunit']
        else:
            rain_xcd = []
            rain_ycd = []
            rain_coordsys = []
        for j in range(len(rain_xcd)):
            if rain_coordsys[j] == 'degrees':
                unit=u.deg
                frm = frames.HeliographicStonyhurst
            elif rain_coordsys[j] == 'arcsec':
                unit=u.arcsec
                frm=frames.Helioprojective
            rain_coord = SkyCoord(
                rain_xcd[j]*unit,
                rain_ycd[j]*unit,
                obstime=tele_start.astype(str),
                observer='earth',
                frame=frm
            )
            rain_coord = rain_coord.transform_to(frames.Helioprojective)
            del_x = float((tele_pointing.Tx - rain_coord.Tx).value)
            del_y = float((tele_pointing.Ty - rain_coord.Ty).value)
            distance = np.sqrt(del_x**2 + del_y**2)
            # Check if it's nearby:
            if distance <= 200:
                event_time = np.datetime64(rain_peak[j].value[0])
                # Best Case. Perfectly in field and on time
                if (distance <= 100):
                    rain_fudge.append(2)
                # Second best case. Slightly out of time, but perfectly in field.
                else:
                    rain_fudge.append(1)
        # Check our rains. If there are any, append them to the event list.
        if len(rain_fudge) > 0:
            event_factors.append(np.nanmax(rain_fudge))
            events.append('Coronal Rain')
    
    if len(event_factors) > 0:
        event_correction = np.nanmax(event_factors)
        event_correction_type = events[event_factors.index(np.nanmax(event_factors))]
    else:
        event_correction = 0
        event_correction_type = 'None'
        
    #### Putting it all together...
    if len(see_cv) == 0:
        see_cv = np.nan
    else:
        see_cv = np.nanmean(np.array(see_cv))
    
    data_quality = {
        'OBSTYPE':obstype,
        'DURATION':duration.astype('timedelta64[m]').astype(int),
        'ZYLA_CV':zyla_cv,
        'CONT_CV':cont_cv,
        'GBAND_CV':gband_cv,
        'CAK_CV':cak_cv,
        'IBIS_CV':ibis_cv,
        'SEE_CV':see_cv,
        'EVENT_FACTOR':event_correction,
        'EVENT_TYPE':event_correction_type
    }
    return data_quality

def update_quality_files(date, obstype):
    quality_params = np.genfromtxt(
            "/sunspot/solardata/quality_control/quality_params.txt", 
            delimiter=',', 
            names=True,
            encoding=None,
            dtype=None
    )
    see_keys = ['ZYLA_CV','CONT_CV','GBAND_CV','CAK_CV','IBIS_CV','SEE_CV']
    percentile_levels = np.arange(0,100,1)

    # Do the full archive first...
    with open("/sunspot/solardata/quality_control/archive_quality.txt","r") as aq:
        archive_dates = aq.readlines()[1:]
    archive_dates = [i.split(",")[0] for i in archive_dates]
    if date not in archive_dates:
        idx = list(quality_params['DATE']).index(date)
        seeing_percentiles = []
        for key in see_keys:
            if not np.isnan(quality_params[key][idx]):
                full_cv = quality_params[key][~np.isnan(quality_params[key])]
                percentiles = np.percentile(full_cv, percentile_levels)
                seeing_percentiles.append(
                        100 - percentile_levels[
                            np.argwhere(quality_params[key][idx] >= percentiles).max()
                        ]
                )
        if len(seeing_percentiles) > 0:
            seeing_factor = round(np.nanmean(seeing_percentiles)/10,1)
        else:
            seeing_factor = np.nan
        # FUCK MY SEEING PERCENTILES ARE BACKWARDS
        # Okay. I "fixed" it. It's 100 - the percentile index, since lower is better here...
        duration_percentile = []
        if not np.isnan(quality_params['DURATION'][idx]):
            full_dur = quality_params['DURATION'][~np.isnan(quality_params['DURATION'])]
            percentiles = np.percentile(full_dur, percentile_levels)
            duration_percentile.append(
                    percentile_levels[
                        np.argwhere(quality_params['DURATION'][idx] >= percentiles).max()
                    ]
            )
        if len(duration_percentile) > 0:
            duration_factor = duration_percentile[0]/10
        else:
            duration_factor = np.nan
        if (not np.isnan(seeing_factor)) & (not np.isnan(duration_factor)):
            quality = round(np.nanmean([seeing_factor, duration_factor]) + quality_params['EVENT_FACTOR'][idx], 2)
        else:
            quality = np.nan
        write_list = [
                quality_params['DATE'][idx],
                quality_params['OBSTYPE'][idx],
                str(quality),
                str(seeing_factor),
                str(duration_factor),
                str(quality_params['EVENT_FACTOR'][idx]),
                quality_params['EVENT_TYPE'][idx]
        ]
        write_str = ",".join(write_list)+ "\n"

        with open("/sunspot/solardata/quality_control/archive_quality.txt","a") as quals:
            quals.write(write_str)
    
    # Now we do the observing series-specific qualities. First have to check that it's an allowed obstype:
    allowed_obs = ['flare','filament','qs','psp']
    if obstype in allowed_obs:
        with open("/sunspot/solardata/quality_control/"+obstype+"_quality.txt","r") as obs_qc:
            obs_dates = obs_qc.readlines()[1:]
        obs_dates = [i.split(",")[0] for i in obs_dates]
        if date not in obs_dates:
            idx = list(quality_params['DATE']).index(date)
            seeing_percentiles = []
            for key in see_keys:
                if not np.isnan(quality_params[key][idx]):
                    full_cv = quality_params[key][quality_params['OBSTYPE'] == obstype]
                    full_cv = full_cv[~np.isnan(full_cv)]
                    percentiles = np.percentile(full_cv, percentile_levels)
                    seeing_percentiles.append(
                            100 - percentile_levels[
                                np.argwhere(quality_params[key][idx] >= percentiles).max()
                            ]
                    )
            if len(seeing_percentiles) > 0:
                seeing_factor = round(np.nanmean(seeing_percentiles)/10,1)
            else:
                seeing_factor = np.nan
            duration_percentile = []
            if not np.isnan(quality_params['DURATION'][idx]):
                full_dur = quality_params['DURATION'][quality_params['OBSTYPE'] == obstype]
                full_dur = full_dur[~np.isnan(full_dur)]
                percentiles = np.percentile(full_dur, percentile_levels)
                duration_percentile.append(
                        percentile_levels[
                            np.argwhere(quality_params['DURATION'][idx] >= percentiles).max()
                        ]
                )
            if len(duration_percentile) > 0:
                duration_factor = duration_percentile[0]/10
            else:
                duration_factor = np.nan
            if (not np.isnan(seeing_factor)) & (not np.isnan(duration_factor)):
                quality = round(np.nanmean([seeing_factor, duration_factor]) + quality_params['EVENT_FACTOR'][idx], 2)
            else:
                quality = np.nan
            write_list = [
                    quality_params['DATE'][idx],
                    quality_params['OBSTYPE'][idx],
                    str(quality),
                    str(seeing_factor),
                    str(duration_factor),
                    str(quality_params['EVENT_FACTOR'][idx]),
                    quality_params['EVENT_TYPE'][idx]
            ]
            write_str = ",".join(write_list)+ "\n"

            with open("/sunspot/solardata/quality_control/"+obstype+"_quality.txt","a") as obsqual:
                obsqual.write(write_str)
    return

def auto_update_dq_files(indir):
    dirlist = indir.split("/")
    date = "-".join([x for x in dirlist if x.isnumeric()])
    try:
        quality_params = determine_quality_metaparams(indir, date)
    except:
        quality_params = None
    # Check if this date is in the archive quality file, as well as the specific file for its obstype
    # Only open the files for appending if the date isn't in the files already
    with open("/sunspot/solardata/quality_control/quality_params.txt","r") as afile:
        archive_dates = afile.readlines()[1:]
    archive_dates = [i.split(",")[0] for i in archive_dates]
    if date not in archive_dates:
        if quality_params:
            write_list = [date]
            for key in list(quality_params.keys()):
                write_list.append(str(quality_params[key]))
            write_str = ",".join(write_list) + "\n"
        else:
            write_str = date+',nan,nan,nan,nan,nan,nan,nan,nan,nan,nan\n'
        with open('/sunspot/solardata/quality_control/quality_params.txt','a') as afile:
            afile.write(write_str)
    if quality_params:
        update_quality_files(date, quality_params['OBSTYPE'])
    return

    

