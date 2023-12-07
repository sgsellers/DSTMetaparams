import numpy as np
import os
import glob
from airium import Airium


def search_for_rosa_data(indir):
    """Searches input directory for ROSA data in level-0, level-1, or level-1.5.
    Returns two lists: one of each filter, the other of its corresponding reduction level.
    """
    l0 = os.path.join(indir, 'level0')
    l1 = os.path.join(indir, 'level1')
    wvls = ['gband','4170','cak','3500']
    formatted = ['G-band','4170', 'Ca K', '3500']
    filters_exist = []
    highest_level = []
    for i in range(len(wvls)):
        l0search = glob.glob(l0 + "/**/*"+wvls[i]+"*/",recursive=True)
        if len(l0search) > 0:
            filters_exist.append(formatted[i])
            l1search = glob.glob(l1 + "/**/"+wvls[i],recursive=True)
            if len(l1search) > 0:
                pspkl = l1search[0]+"/**/postSpeckle"
                pdstr = l1search[0]+"/**/splineDestretch"
                if len(glob.glob(pdstr + "/*.fits",recursive=True)) > 0:
                    highest_level.append("1.5")
                elif len(glob.glob(pspkl + "/*.fits",recursive=True)) > 0:
                    highest_level.append("1")
                else:
                    highest_level.append("0")
            else:
                highest_level.append("0")
    return filters_exist, highest_level

def search_for_zyla_data(indir):
    """Search input directory for ROSA data in level-0, 1, or 1.5. 
    Returns a list which is either empty, or a single str entry.
    The entry is the data level
    """
    l0 = os.path.join(indir, "level0")
    l1 = os.path.join(indir, "level1")
    l0search = glob.glob(l0 + "/**/*yla*/",recursive=True)
    highest_level = []
    if len(l0search) > 0:
        l1search = glob.glob(l1 + "/**/zyla/", recursive=True)
        if len(l1search) > 0:
            pspkl = l1search[0]+ "/**/postSpeckle"
            pdstr = l1search[0]+ "/**/splineDestretch"
            if len(glob.glob(pdstr + "/*.fits",recursive=True)) > 0:
                highest_level.append("1.5")
            elif len(glob.glob(pspkl + "/*.fits",recursive=True)) > 0:
                highest_level.append("1")
            else:
                highest_level.append("0")
        else:
            highest_level.append("0")
    return highest_level

def search_for_ibis_data(indir):
    l0 = os.path.join(indir,'level0')
    l1 = os.path.join(indir, 'level1')
    l0search = glob.glob(l0 + "/**/*ibis*/", recursive=True)
    highest_level = []
    if len(l0search) > 0:
        l1search = glob.glob(l1 +"/**/ibis/", recursive=True)
        if len(l1search) > 0:
            highest_level.append("1")
        else:
            highest_level.append("0")
    return highest_level


def search_for_firs_data(indir):
    """Search input directory for FIRS data in level-0, 1, or 1.5. 
    Returns two lists, one with the instrument (firs proper or SJI), the other with data levels.
    """
    l0 = os.path.join(indir, "level0")
    l1 = os.path.join(indir, "level1")
    l2 = os.path.join(indir, "level2")
    instr = []
    highest_level = []
    sji_search = glob.glob(l0 + "/**/*firs*slitjaw*/", recursive=True)
    if len(sji_search) > 0:
        instr.append("Slitjaw")
        highest_level.append("0")
    l0search = glob.glob(l0 + "/**/*firs*/",recursive=True)
    l0search = [f for f in l0search if "slitjaw" not in f]
    if len(l0search) > 0:
        instr.append("10830")
        l2search = glob.glob(l2 + "/**/*firs*",recursive=True)
        if len(l2search) > 0:
            highest_level.append("2")
        else:
            l15search = glob.glob(l1 + "/*firs*/*.fits")
            l1search = glob.glob(l1 + "/*firs*/*.sav")
            if len(l15search) > 0:
                highest_level.append("1.5")
            elif len(l1search) > 0:
                highest_level.append("1")
            else:
                highest_level.append("0")
    return instr, highest_level

def search_for_spinor_data(indir):
    """Search input directory for SPINOR data. Returns list with wavelength and another with level"""
    l0 = os.path.join(indir, 'level0')
    l1 = os.path.join(indir, 'level1')
    wvls = ['6302','8542','5876','slitjaw']
    filters_exist = []
    highest_level = []
    for i in wvls:
        l0search = glob.glob(l0 + "/**/*spinor*"+i+"*/",recursive=True)
        if len(l0search) > 0:
            filters_exist.append(i)
            l1search = glob.glob(l1 + "/**/spinor/"+i,recursive=True)
            if len(l1search) > 0:
                highest_level.append("1")
            else:
                highest_level.append("0")
    return filters_exist, highest_level

def search_for_hsg_data(indir):
    l0 = os.path.join(indir, 'level0')
    l1 = os.path.join(indir, 'level1')
    wvls = ['6302','8542','5876','5893','5896','slitjaw']
    filters_exist = []
    highest_level = []
    for i in wvls:
        l0search = glob.glob(l0 + "/**/*hsg*"+i+"*/",recursive=True)
        if len(l0search) > 0:
            filters_exist.append(i)
            l1search = glob.glob(l1 + "/**/hsg/"+i,recursive=True)
            if len(l1search) > 0:
                highest_level.append("1")
            else:
                highest_level.append("0")
    return filters_exist, highest_level

def search_for_data_levels(indir):
    """Searches input directory for instruments that are level0, level1, and level1.5.
    Returns a dictionary of instrument names and corresponding data levels
    """
    rosa_f, rosa_l = search_for_rosa_data(indir)
    zyla_l = search_for_zyla_data(indir)
    ibis_l = search_for_ibis_data(indir)
    firs_f, firs_l = search_for_firs_data(indir)
    spinor_f, spinor_l = search_for_spinor_data(indir)
    hsg_f, hsg_l = search_for_hsg_data(indir)

    aa = "&#8491;"

    instr_names = []
    instr_lvls = []
    for i in range(len(rosa_f)):
        if rosa_f[i].isnumeric():
            instr_names.append("ROSA " + rosa_f[i] + aa)
        else:
            instr_names.append("ROSA " + rosa_f[i])
        instr_lvls.append("Level-" + rosa_l[i]) 
    for i in range(len(zyla_l)):
        instr_names.append("Zyla 6563"+aa)
        instr_lvls.append("Level-" + zyla_l[i])
    for i in range(len(firs_f)):
        if firs_f[i] == "10830":
            instr_names.append("FIRS 10830"+aa)
        else:
            instr_names.append("FIRS "+firs_f[i])
        instr_lvls.append("Level-" + firs_l[i])
    for i in range(len(spinor_f)):
        if spinor_f[i] != 'slitjaw':
            instr_names.append("SPINOR " + spinor_f[i] + aa)
        else:
            instr_names.append("SPINOR " + spinor_f[i])
        instr_lvls.append("Level-" + spinor_l[i])
    for i in range(len(hsg_f)):
        if hsg_f[i] != 'slitjaw':
            instr_names.append("HSG " + hsg_f[i] + aa)
        else:
            instr_names.append("HSG " + hsg_f[i])
        instr_lvls.append("Level-" + hsg_l[i])
    for i in range(len(ibis_l)):
        instr_names.append("IBIS Scans")
        instr_lvls.append("Level-" + ibis_l[i])
    return instr_names, instr_lvls

def automatic(indir):
    if indir[-1] != "/":
        indir += "/"
    instrument_names, instrument_levels = search_for_data_levels(indir)
    list_of_cak_seeing = sorted(glob.glob(indir + "*cak_seeing_quicklook*.png"))
    list_of_4170_seeing = sorted(glob.glob(indir + "*4170_seeing_quicklook*.png"))
    list_of_gband_seeing = sorted(glob.glob(indir + "*gband_seeing_quicklook*.png"))
    list_of_3500_seeing = sorted(glob.glob(indir + "*3500_seeing_quicklook*.png"))
    list_of_zyla_seeing = sorted(glob.glob(indir + "*zyla_seeing_quicklook*.png"))
    list_of_ibis_seeing = sorted(glob.glob(indir + "*ibis_seeing_quicklook*.png"))

    nobs = np.nanmax(np.array([len(list_of_cak_seeing),len(list_of_4170_seeing),len(list_of_gband_seeing),len(list_of_3500_seeing),len(list_of_zyla_seeing), len(list_of_ibis_seeing)],dtype=int))

    date = "-".join(indir.split("/")[-4:-1])

    observing_summary = sorted(glob.glob(indir + "*observing_summary*"))
    if len(observing_summary) > 0:
        observing_summary = observing_summary[0].replace("/sunspot","")
    context_image = sorted(glob.glob(os.path.join(indir, "context_ims") + "/*context_image*"))
    if len(context_image) > 0:
        context_image = context_image[0].replace("/sunspot","")

    css = "/css/nmsu_ssoc.css"
    lenses = "/css/SSOC_lenses.png"
    obslogs = sorted(glob.glob(os.path.join(indir,"**") + "/*obs*log*/", recursive=True))[0]
    
    # Getting the type of observations and fancing it up.

    obsreg = open("/var/www/html/observations_registry.csv","r")
    oreg_lines = obsreg.readlines()
    obsreg.close()
    all_datestrs = [l.split(",")[1] for l in oreg_lines]
    all_otypes = [l.split(",")[2] for l in oreg_lines]

    abbrv_obstype = all_otypes[all_datestrs.index(date)]
    allowed_obstypes = ['filament','flare','limb','other','pi','polcal','psp','qs','test']
    full_obstypes = ['Filament Observation','Active Region Observation',
                     'Limb Observation','Uncatagorized Observation','PI-led Observation',
                     'Polcal Only','Parker Solar Probe Support',
                     'Quiet Sun Observation','Test Series']
    observation_type = full_obstypes[allowed_obstypes.index(abbrv_obstype)]

    # Finding the next/previous date with data.
    date_dirs = sorted(glob.glob("/sunspot/solardata/2*/*/*/"))
    obsdate_index = date_dirs.index(indir)
    if obsdate_index == 0:
        prevDate = None
    else:
        prevDate = date_dirs[obsdate_index - 1]
    if obsdate_index == len(date_dirs)-1:
        nextDate = None
    else:
        nextDate = date_dirs[obsdate_index + 1]

    monthPath = os.path.join(os.path.join("/solardata",date.split('-')[0]),date.split('-')[1])

    qp_obstypes = ['filament','flare','psp','qs']
    path_to_quality_params = "/sunspot/solardata/quality_control/"
    qual_files = []
    if abbrv_obstype in qp_obstypes:
        qual_files.append(os.path.join(path_to_quality_params, abbrv_obstype + "_quality.txt"))
    qual_files.append(os.path.join(path_to_quality_params, "archive_quality.txt"))
    quality_param = []
    seeing_param = []
    duration_param = []
    event = []
    for file in qual_files:
        quality_array = np.genfromtxt(file, delimiter=",", names=True, encoding=None, dtype=None)
        try:
            date_idx = list(quality_array['DATE']).index(date)
            if not np.isnan(quality_array['QUALITY'][date_idx]):
                quality_param.append(quality_array['QUALITY'][date_idx])
                seeing_param.append(int(quality_array['SEEING_FACTOR'][date_idx]*10))
                duration_param.append(int(quality_array['DURATION_FACTOR'][date_idx]*10))
                etype = quality_array['EVENT_TYPE'][date_idx]
                eparam = quality_array['EVENT_FACTOR'][date_idx]
                if etype == "None":
                    event.append("No events in the HEK database")
                elif etype[0] in ["X","M","C","B"]:
                    if eparam == 0.5:
                        event.append("The HEK database shows a GOES " + etype + " flare just outside the observing window or FOV")
                    elif eparam == 1.0:
                        event.append("The HEK database shows partial coverage of a GOES "+etype+" flare")
                    elif eparam == 2.0:
                        event.append("The HEK database shows full coverage of a GOES "+etype+" flare")
                else:
                    if eparam == 0.5:
                        event.append("The HEK database shows a "+etype+" event just outside the observing window or FOV")
                    elif eparam == 1.0:
                        event.append("The HEK database shows partial coverage of a "+etype+" event")
                    else:
                        event.append("The HEK database shows full coverage of a "+etype+" event")
        except:
            pass
    a = Airium()

    a('<!DOCTYPE HTML>')
    with a.html():
        with a.head():
            a.title(_t=date)
            a.meta(charset="utf-8")
            a.link(rel='stylesheet', href=css)
        with a.body():
            # First, we have to build the sidebar...
            with a.div(klass="wrapper"):
                with a.div(klass="sidebar"):
                    with a.div(klass="profile"):
                        a.img(src=lenses, alt="Lenses at the DST, image from Sean Sellers")
                        with a.b():
                            a(date + " Data Overview")
                        with a.ul():
                            with a.li():
                                with a.a(href=monthPath):
                                    with a.b():
                                        a("Monthly Overview")
                            with a.li():
                                with a.a(href=os.path.join(indir,"level0").replace("/sunspot","")):
                                    with a.b():
                                        a("Level-0 Data")
                            with a.li():
                                with a.a(href=os.path.join(indir,"level1").replace("/sunspot","")):
                                    with a.b():
                                        a("Level-1 Data")
                            if any("2" in entry for entry in instrument_levels):
                                with a.li():
                                    with a.a(href=os.path.join(indir,"level2").replace("/sunspot","")):
                                        with a.b():
                                            a("Level-2 Data")
                            with a.li():
                                with a.a(href=obslogs.replace("/sunspot","")):
                                    with a.b():
                                        a("Observing Logs")
                        a.hr()
                        with a.ul():
                            with a.li():
                                with a.a(href="#datasum"):
                                    a.b(_t="Data Summary")
                            with a.li():
                                with a.a(href="#obssum"):
                                    a.b(_t="Observation Summary")
                            if len(list_of_4170_seeing) > 0:
                                with a.li():
                                    with a.a(href="#4170_seeing"):
                                        a.b(_t="4170&#8491; Seeing Summary")
                            if len(list_of_gband_seeing) > 0:
                                with a.li():
                                    with a.a(href="#gband_seeing"):
                                        a.b(_t="G-band Seeing Summary")
                            if len(list_of_cak_seeing) > 0:
                                with a.li():
                                    with a.a(href="#cak_seeing"):
                                        a.b(_t="Ca K Seeing Summary")
                            if len(list_of_3500_seeing) > 0:
                                with a.li():
                                    with a.a(href="#3500_seeing"):
                                        a.b(_t="3500&#8491; Seeing Summary")
                            if len(list_of_zyla_seeing) > 0:
                                with a.li():
                                    with a.a(href="#zyla_seeing"):
                                        a.b(_t="H&alpha; 6563&#8491; Seeing Summary")
                            if len(list_of_ibis_seeing) > 0:
                                with a.li():
                                    with a.a(href="#ibis_seeing"):
                                        a.b(_t="IBIS Seeing Summary")
            # Sidebar is wrapped up. On to the body.
            # First, simple header...
            with a.section(id="datasum"):
                with a.div(klass="overview"):
                    with a.ul():
                        if prevDate:
                            with a.li(klass="prev"):
                                with a.a(href=prevDate.replace("/sunspot","")):
                                    a("&#10094;")
                        if nextDate:
                            with a.li(klass="next"):
                                with a.a(href=nextDate.replace("/sunspot","")):
                                    a("&#10095;")
                        with a.li(klass="title"):
                            a(date + ": " + observation_type)
                            a.br()
                            with a.span(klass="monthqp-wrapper"):
                                with a.a(href=monthPath):
                                    a("Monthly Overview")
                                if len(quality_param) != 0:
                                    with a.p():
                                        a("&nbsp;|&nbsp;")
                                    with a.div(klass="popup"):
                                        with a.p():
                                            a("Data Quality: "+str(quality_param[0]))
                                        with a.span(klass="popuptext", id="dqPop"):
                                            if len(quality_param) == 1:
                                                with a.u():
                                                    a(str(quality_param[0]) + " is calculated from the full data archive:")
                                                with a.ul():
                                                    with a.li():
                                                        a("Top "+str(seeing_param[0])+"&percnt; by seeing stability")
                                                    with a.li():
                                                        a("Top "+str(duration_param[0])+"&percnt; by duration of observations")
                                                    with a.li():
                                                        a(event[0])
                                            else:
                                                with a.u():
                                                    a(str(quality_param[0])+" is calculated from all "+observation_type+" datasets:")
                                                with a.ul():
                                                    with a.li():
                                                        a("Top "+str(seeing_param[0])+"&percnt; by seeing stability")
                                                    with a.li():
                                                        a("Top "+str(duration_param[0])+"&percnt; by duration of observations")
                                                    with a.li():
                                                        a(event[0])
                                                with a.u():
                                                    a("When the full archive is considered, the quality is "+str(quality_param[1])+":")
                                                with a.ul():
                                                    with a.li():
                                                        a("Top "+str(seeing_param[1])+"&percnt; by seeing stability")
                                                    with a.li():
                                                        a("Top "+str(duration_param[1])+"&percnt; by duration of observations")
                                                    with a.li():
                                                        a(event[1])


            # Pointing information, data levels, and an explanation thereof.
            with a.section():
                with a.div(klass="container"):
                    if type(context_image) is str:
                        with a.header(klass="major"):
                            a.h2(_t="Pointing Information:")
                        a.img(src=context_image, alt="Observing Pointing Summary", width="100%", klass="center")
                    with a.header(klass="major"):
                        a.h2(_t="Instruments Used:")
                    with a.table():
                        with a.tr():
                            a.th(_t="Instrument")
                            a.th(_t="Reduction Status")
                        for i in range(len(instrument_names)):
                            with a.tr():
                                a.td(_t=instrument_names[i])
                                a.td(_t=instrument_levels[i])
                    with a.header(klass="major"):
                        a.h3(_t="Note on Data Reduction Levels:")
                    a.p(_t="Data levels are defined as follows:")
                    with a.ul():
                        with a.li():
                            a.b(_t="Level-0:")
                            a("Raw data from the telescope. Image data at level 0 have not been gain corrected. Spectropolarimetric modulation states have not been separated.")
                        with a.li():
                            a.b(_t="Level-1:")
                            a("First order corrections have been applied. Imaging data has been gain-corrected and processed via speckle burst reconstruction. Spectropolarimetric data is separated into Stokes I, Q, U, and V, but significant cleaning and post-processing has not taken place. Files at level 1 may be in IDL .sav format, or other nonstandard data formats. Note that imaging data at this step may or may not have been processed via a non-flow-preserving destretch algorithm. The individual filenames will indicate whether or not this is the case.")
                        with a.li():
                            a.b(_t="Level-1.5:")
                            a("Further post-processing has taken place. Data at level-1.5 can be considered science-ready. Imaging data has been destretched using both a method designed to preserve surface flows, as well as a method that is more stable under variable seeing conditions. Spectropolarimetric and spectroscopic data have been wavelength-calibrated and corrected for fringing. Data at level-1.5 have been repacked as FITS files")
                        with a.li():
                            a.b(_t="Level-2:")
                            a("Level-2 data have been processed to display physical parameters, such as magnetic field data, or other inversion results. Data at level-2 may have various filetypes.")
            # Observing Summary Plot
            with a.section(id="obssum"):
                with a.div(klass="container"):
                    if type(observing_summary) is str:
                        with a.header(klass="major"):
                            a.h2(_t="Observation Series Summary:")
                        a.p(_t="There were "+str(nobs)+" different observing series on this date. Observing and calibration windows covered the following ranges:")
                        a.img(src=observing_summary, alt="Observing Times Summary", width="75%", klass="center")

            # Seeing Parameter Plots!
            with a.section(id="seeing"):
                with a.div(klass="container"):
                    with a.header(klass="major"):
                        a.h2(_t="Seeing Parameters by Pointing and Passband:")
                    with a.p():
                        a("Seeing parameters are calculated from Level-0, unprocessed data, before speckle reconstruction. The \"median\" and +/- 1-sigma images below are calculated from the Helmli-Scherer mean (HSM) for ROSA, and the Median Filter Gradiant Similarity (MFGS) metric for Zyla. These metrics (and others, such as the RMS, and mean of the image) are calculated for the full observing day. HSM and MFGS are the most sensitive to unknown atmospheric conditions, however, the HSM is faster and tends to perform better with photospheric data, while MFGS performs better with chromospheric structures.")
                    # 4170
                    if len(list_of_4170_seeing) > 0:
                        with a.header(klass="major", id="4170_seeing"):
                            a.h3(_t="4170&#8491; Seeing Plots")
                    for i in range(len(list_of_4170_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="4170&#8491; Pointing "+str(i+1)+" of "+str(len(list_of_4170_seeing)))
                        a.img(
                                src=list_of_4170_seeing[i].replace("/sunspot",""), 
                                alt="4170 Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
                    # G-band
                    if len(list_of_gband_seeing) > 0:
                        with a.header(klass="major", id="gband_seeing"):
                            a.h3(_t="G-band Seeing Plots")
                    for i in range(len(list_of_gband_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="G-band Pointing "+str(i+1)+" of "+str(len(list_of_gband_seeing)))
                        a.img(
                                src=list_of_gband_seeing[i].replace("/sunspot",""), 
                                alt="G-band Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
                    # Ca K
                    if len(list_of_cak_seeing) > 0:
                        with a.header(klass="major", id="cak_seeing"):
                            a.h3(_t="Ca K Seeing Plots")
                    for i in range(len(list_of_cak_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="Ca K Pointing "+str(i+1)+" of "+str(len(list_of_cak_seeing)))
                        a.img(
                                src=list_of_cak_seeing[i].replace("/sunspot",""), 
                                alt="Ca K Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
                    # 3500
                    if len(list_of_3500_seeing) > 0:
                        with a.header(klass="major", id="3500_seeing"):
                            a.h3(_t="3500&#8491; Seeing Plots")
                    for i in range(len(list_of_3500_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="3500&#8491; Pointing "+str(i+1)+" of "+str(len(list_of_3500_seeing)))
                        a.img(
                                src=list_of_3500_seeing[i].replace("/sunspot",""), 
                                alt="3500 Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
                    # Zyla Ha
                    if len(list_of_zyla_seeing) > 0:
                        with a.header(klass="major", id="zyla_seeing"):
                            a.h3(_t="H&alpha; 6563&#8491; Seeing Plots")
                    for i in range(len(list_of_zyla_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="H&alpha; 6563&#8491; Pointing "+str(i+1)+" of "+str(len(list_of_zyla_seeing)))
                        a.img(
                                src=list_of_zyla_seeing[i].replace("/sunspot",""), 
                                alt="6563 Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
                    # IBIS
                    if len(list_of_ibis_seeing) > 0:
                        with a.header(klass='major', id='ibis_seeing'):
                            a.h3(_t="IBIS Seeing Plots")
                    for i in range(len(list_of_ibis_seeing)):
                        with a.header(klass="minor"):
                            a.h4(_t="IBIS Pointing " + str(i+1) + " of " + str(len(list_of_ibis_seeing)))
                        a.img(
                                src=list_of_ibis_seeing[i].replace("/sunspot",""),
                                alt="IBIS Pointing "+str(i+1),
                                width="75%",
                                klass="center"
                        )
            a.script(src="/css/daily_qualpop.js")
    
    with open(indir + "index.html","wb") as f:
        f.write(bytes(a))


