import numpy as np
import os
import glob
import datetime
from airium import Airium
from calendar import monthrange

def automatic(indir):
    if indir[-1] == "/":
        indir = indir[:-1]
# Integer days with observations. Subtract one because months start at 0
# Will have to add one to make paths later
    days_with_obs = [int(os.path.split(path)[-1]) - 1 for path in sorted(glob.glob(os.path.join(indir,"*"))) if os.path.isdir(path) and len(glob.glob(os.path.join(path,"*"))) > 0]

    year = os.path.split(os.path.split(indir)[0])[1]
    month = os.path.split(indir)[1]

    yearpath = os.path.join("/sunspot/solardata",year)
    months_in_year = sorted(glob.glob(os.path.join(yearpath,"*")))

    monthidx = months_in_year.index(indir)
    if monthidx == 0:
        prevyear = os.path.join("/sunspot/solardata", str(int(year) - 1))
        prevmonths_in_year = sorted(glob.glob(os.path.join(prevyear,"*")))
        if len(prevmonths_in_year) == 0:
            prevmonth = None
        else:
            prevmonth = prevmonths_in_year[-1]

    else:
        prevmonth = months_in_year[monthidx-1]

    if monthidx == len(months_in_year) - 1:
        nextyear = os.path.join("/sunspot/solardata", str(int(year) + 1))
        nextmonths_in_year = sorted(glob.glob(os.path.join(nextyear,"*")))
        if len(nextmonths_in_year) == 0:
            nextmonth = None
        else:
            nextmonth = nextmonths_in_year[0]
    else:
        nextmonth = months_in_year[monthidx+1]

    monthname = ['January', 'February', 
                 'March', 'April', 
                 'May', 'June', 
                 'July', 'August', 
                 'September', 'October', 
                 'November', 'December']

    weekdays = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]

    datestrs_with_obs = [year + "-" + month + "-" + str(int(dy) + 1).zfill(2) for dy in days_with_obs]
    path_to_obsreg = "/var/www/html/observations_registry.csv"
    obstype = []
    obsreg = open(path_to_obsreg,"r").readlines()
    for line in obsreg:
        for i in range(len(datestrs_with_obs)):
            if datestrs_with_obs[i] in line:
                obstype.append(line.split(",")[2].lower())

    allowed_obstypes = ['flare','filament','other','psp','polcal','pi','qs','limb','test']
    obstype_names = ['Active Region Study', 'Filament Observation', 'Other Observation',
                     'Parker Solar Probe Support', 'Polcal Only', 'PI-led study',
                     'Quiet Sun Study', 'Limb Study',"Instrument Tests"]
    css_obstypes = ['ar','fil','other','psp','polcal','pi','qs','limb',"test"]
#Week starts with Monday at 0
    starting_day, days_in_month = monthrange(int(year), int(month))

    qp_obstypes = ['filament','flare','psp','qs']
    path_to_quality_params = "/sunspot/solardata/quality_control/"
    qual_files = []
    for i in range(len(obstype)):
        if obstype[i] in qp_obstypes:
            qual_files.append(os.path.join(path_to_quality_params, obstype[i] + "_quality.txt"))
        else:
            qual_files.append(os.path.join(path_to_quality_params, "archive_quality.txt"))
    quality_param = []
    seeing_param = []
    duration_param = []
    event = []
    qual_type = []
    for i in range(len(datestrs_with_obs)):
        date = datestrs_with_obs[i]
        quality_array = np.genfromtxt(qual_files[i], delimiter=",", names=True, encoding=None, dtype=None)
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
            else:
                quality_param.append("nan")
                seeing_param.append("nan")
                duration_param.append("nan")
                event.append("nan")
        except:
            quality_param.append("nan")
            seeing_param.append("nan")
            duration_param.append("nan")
            event.append("nan")


        if obstype[i] in qp_obstypes:
            formal_name = obstype_names[allowed_obstypes.index(obstype[i])]
            full_str = str(quality_param[-1]) + " is calculated from all "+formal_name+" datasets:"
            qual_type.append(full_str)
        else:
            full_str = str(quality_param[-1]) + " is caculated from the full data archive:"
            qual_type.append(full_str)

    css = "/css/nmsu_months.css"
    lenses = "/css/SSOC_lenses.png"
    a = Airium()

    a("<!DOCTYPE HTML>")

    modal_ids = []


    with a.html():
        with a.head():
            a.title(_t=monthname[int(month)-1] + " " + year)
            a.meta(charset="utf-8")
            a.link(rel='stylesheet', href=css)
        with a.body():
            with a.div(klass="wrapper"):
                with a.div(klass="sidebar"):
                    with a.div(klass="profile"):
                        a.img(src=lenses, alt="DST Lenses, Image courtesy of Sean Sellers")
                        with a.b():
                            a(monthname[int(month) - 1] + " " + year + " Overview")
                        a.hr()
                        with a.ul():
                            for i in range(len(days_with_obs)):
                                with a.li():
                                    with a.a(href=str(days_with_obs[i]+1).zfill(2)):
                                        with a.b():
                                            if quality_param[i] != "nan":
                                                a(monthname[int(month) - 1] + " " + str(days_with_obs[i] + 1).zfill(2) + ", " + year+": Quality "+str(quality_param[i])+"/10")
                                            else:
                                                a(monthname[int(month) - 1] + " " + str(days_with_obs[i] + 1).zfill(2) + ", " + year)
                ####
                with a.div(klass="container"):
                    a.h1("Monthly Overview: " + monthname[int(month) - 1] + " " + year)
                    with a.div(klass="month"):
                        with a.ul():
                            if prevmonth:
                                with a.li(klass="prev"):
                                    with a.a(href=prevmonth.replace("/sunspot","")):
                                        a("&#10094;")
                            if nextmonth:
                                with a.li(klass="next"):
                                    with a.a(href=nextmonth.replace("/sunspot","")):
                                        a("&#10095;")
                            with a.li():
                                a(monthname[int(month) - 1])
                                a.br()
                                with a.span(style="font-size:18px"):
                                    a(year)
                    with a.div(klass="calendar-wrapper"):
                        with a.div(klass="weekdays"):
                            with a.ul():
                                for i in weekdays:
                                    with a.li():
                                        a(i)
                        with a.div(klass="day-grid"):
                            with a.ul():
                                for i in range(starting_day):
                                    with a.li():
                                        a(" ")
                                for i in range(days_in_month):
                                    if i not in days_with_obs:
                                        with a.li():
                                            a(str(i + 1))
                                    else:
                                        index = days_with_obs.index(i)
                                        otype = obstype[index]
                                        if otype not in allowed_obstypes:
                                            otype = "other"
                                        css_otype = css_obstypes[allowed_obstypes.index(otype)]

                                        with a.li():
                                            with a.button(klass=css_otype, 
                                                          id=str(i+1).zfill(2), 
                                                          href="#m"+str(i+1).zfill(2)):
                                                a(str(i+1))
                                                modal_ids.append("m" + str(i+1).zfill(2))
                    a.hr()
                    with a.div(klass="key"):
                        with a.ul():
                            for i in range(len(set(obstype))):
                                otype = list(set(obstype))[i]
                                if list(set(obstype))[i] not in allowed_obstypes:
                                    otype = 'other'
                                index = allowed_obstypes.index(otype)
                                pretty_obstype = obstype_names[index]
                                css_otype = css_obstypes[index]
                                with a.li(klass="key-"+css_otype):
                                    a(pretty_obstype)
                for i in range(len(modal_ids)):
                    with a.div(klass="modal", id=modal_ids[i]):
                        with a.div(klass="modal-content"):
                            with a.div(klass="modal-header"):
                                with a.span(klass="close"):
                                    a("&times;")
                                with a.span(klass="monthqp-wrapper"):
                                    with a.a(href=modal_ids[i].replace("m","")):
                                        a(monthname[int(month) - 1] + " " + str(days_with_obs[i] + 1).zfill(2) + ", " + year)
                                    if quality_param[i] != 'nan':
                                        with a.p():
                                            a("&nbsp;|&nbsp;")
                                        with a.div(klass="popup"):
                                            with a.p():
                                                a("Data Quality "+str(quality_param[i]))
                                            with a.span(klass="popuptext", id=modal_ids[i].replace("m","dqPop")):
                                                with a.u():
                                                    a(qual_type[i])
                                                with a.ul():
                                                    with a.li():
                                                        a("Top "+str(seeing_param[i])+"&percnt; by seeing stability")
                                                    with a.li():
                                                        a("Top "+str(duration_param[i])+"&percnt; by duration of observations")
                                                    with a.li():
                                                        a(event[i])
                                            
                            with a.div(klass="modal-body"):
                                context_im = os.path.join(
                                        os.path.join(
                                            modal_ids[i].replace("m",""),
                                            "context_ims"
                                        ),
                                        "context_image.jpg")
                                obs_im = os.path.join(
                                        modal_ids[i].replace("m",""),
                                        "observing_summary.jpg")
                                if os.path.isfile(os.path.join(indir,context_im)) and os.path.isfile(os.path.join(indir,obs_im)):
                                    with a.div(klass="row"):
                                        with a.div(klass="column"):
                                            a.img(src=context_im, alt="Observing Pointing Summary", width="90%", klass="center")
                                        with a.div(klass="column"):
                                            a.img(src=obs_im, alt="Obssum", width="90%", klass="center")
                                elif os.path.isfile(os.path.join(indir,context_im)) and not os.path.isfile(os.path.join(indir,obs_im)):
                                    a.img(src=context_im, alt="Observing Pointing Summary", width="60%",klass="center")
                                elif os.path.isfile(os.path.join(indir,obs_im)) and not os.path.isfile(os.path.join(indir,context_im)):
                                    a.img(src=obs_im, alt="Obssum", width="60%",klass="center")

            a.script(src="/css/modal.js")

    with open(os.path.join(indir,"index.html"),"wb") as f:
        f.write(bytes(a))

