import numpy as np, matplotlib.pyplot as plt, sunpy.map as smap, matplotlib, astropy.units as u, argparse, glob
from sunpy.net import Fido, attrs as a
import os
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames


def automatic(indir, rosa_dxy, zyla_dxy, force_redo = "n"):

    params = {
        'image.origin': 'lower',
        'image.interpolation':'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 300,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'font.size': 12,
        'font.weight':'bold',
        'legend.fontsize': 14,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': False,
        'font.family': 'serif',
        'figure.facecolor': '#fef5e7'
    }

    plt.rcParams.update(params)


    indir = os.path.join(indir,"context_ims")
    savedir = indir
    if force_redo == "n":
        if any("context_image.jpg" in fl for fl in glob.glob(indir + "/*")):
            print("Telescope pointing plot exists, exiting...")
            return
    infile = os.path.join(indir, "pointing_info.txt")
    if not os.path.isfile(infile):
        return
    pointings = np.recfromtxt(infile,delimiter=',',names=True,encoding=None)

    pointings = np.atleast_1d(pointings)

    t_start = np.datetime64(pointings['TIME'][0])
    t_end = np.datetime64(pointings['TIME'][0]) + np.timedelta64(3,'h')

    hmi_list = sorted(glob.glob(savedir + '/*hmi*'))
    chrom_list = sorted(glob.glob(savedir + '/*aia*304*'))
    cor_list = sorted(glob.glob(savedir + '/*aia*193*'))
    flare_query = Fido.search(a.Time((t_start-np.timedelta64(1,"h")).astype(str),(t_end).astype(str)),
                         a.hek.EventType('FL'),a.hek.FL.GOESCls>'B1.0')
    if len(flare_query['hek'])>0:
        flare_xcd = flare_query['hek']['event_coord1']
        flare_ycd = flare_query['hek']['event_coord2']
        flare_peak = flare_query['hek']['event_peaktime']
        flare_coordsys = flare_query['hek']['event_coordunit']
        flare_goescls = flare_query['hek']['fl_goescls']
    else:
        flare_xcd=[]
        flare_ycd=[]
        flare_peak=[]
        flare_coordsys=[]
        flare_goescls=[]

    fe_query = Fido.search(a.Time(t_start.astype('datetime64[D]').astype(str), 
                                   (t_end.astype('datetime64[D]') + np.timedelta64(1,"D")).astype(str)),
                           a.hek.EventType("FE"))
    if len(fe_query['hek']) > 0:
        fe_xcd = fe_query['hek']['event_coord1']
        fe_ycd = fe_query['hek']['event_coord2']
        fe_coordsys = fe_query['hek']['event_coordunit']
    else:
        fe_xcd = []
        fe_ycd = []
        fe_coordsys = []

    cr_query = Fido.search(a.Time(t_start.astype('datetime64[D]').astype(str),
                                  (t_end.astype('datetime64[D]') + np.timedelta64(1,"D")).astype(str)),
                          a.hek.EventType("CR"))
    if len(cr_query['hek']) > 0:
        cr_xcd = cr_query['hek']['event_coord1']
        cr_ycd = cr_query['hek']['event_coord2']
        cr_coordsys = cr_query['hek']['event_coordunit']
    else:
        cr_xcd = []
        cr_ycd = []
        cr_coordsys = []

    if len(hmi_list) == 0:
        cont_search = Fido.search(
                a.Time(t_start.astype(str), t_end.astype(str)),
                a.Instrument.hmi,
                a.Physobs.intensity)
        if len(hmi_list) > 0:
            cont_f = hmi_list
        else:
            cont_f = Fido.fetch(cont_search[0,0],path=savedir)
            while len(cont_f.errors) > 0:
                cont_f = Fido.fetch(cont_f)

        hmi_map = smap.Map(cont_f[0]).rotate(order=3)
    else:
        hmi_map = smap.Map(hmi_list[0]).rotate(order=3)
    
    if len(chrom_list) == 0:
        chrom_search = Fido.search(
                a.Time(t_start.astype(str), t_end.astype(str)),
                a.Instrument.aia,
                a.Wavelength(304*u.angstrom))
        if len(chrom_list) > 0:
            chrom_f = chrom_list
        else:
            chrom_f = Fido.fetch(chrom_search[0,0],path=savedir)
            while len(chrom_f.errors) > 0:
                chrom_f = Fido.fetch(chrom_f)
        chrom_map = smap.Map(chrom_f[0])
    else:
        chrom_map = smap.Map(chrom_list[0])

    if len(cor_list) == 0:
        corona_search = Fido.search(
                a.Time(t_start.astype(str), t_end.astype(str)),
                a.Instrument.aia,
                a.Wavelength(193*u.angstrom))
        if len(cor_list) > 0:
            corona_f = cor_list
        else:
            corona_f = Fido.fetch(corona_search[0,0],path=savedir)
            while len(corona_f.errors) > 0:
                corona_f = Fido.fetch(corona_f)        
        corona_map = smap.Map(corona_f)
    else:
        corona_map = smap.Map(cor_list[0])

    chrom_map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
    corona_map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']

    list_of_rosa_bl = []
    list_of_zyla_bl = []

    for i in range(len(pointings['SLAT'])):
        c = SkyCoord(
                pointings['SLON'][i]*u.deg,
                pointings['SLAT'][i]*u.deg,
                frame=frames.HeliographicStonyhurst,
                observer='earth',
                obstime=pointings['TIME'][i]
        )
        cds = c.transform_to(frames.Helioprojective)
        rtx = float((cds.Tx - (500*rosa_dxy)*u.arcsec).value)
        rty = float((cds.Ty - (500*rosa_dxy)*u.arcsec).value)

        ztx = float((cds.Tx - (1024*zyla_dxy)*u.arcsec).value)
        zty = float((cds.Ty - (1024*zyla_dxy)*u.arcsec).value)

        list_of_rosa_bl.append((rtx,rty))
        list_of_zyla_bl.append((ztx,zty))

    fig = plt.figure(figsize=(18,8))
    ax_hmi = fig.add_subplot(131, projection=hmi_map)
    ax_chr = fig.add_subplot(132, projection=chrom_map)
    ax_cor = fig.add_subplot(133, projection=corona_map)

    hmi_map.plot(axes=ax_hmi)
    chrom_map.plot(axes=ax_chr,clip_interval=(1,99.99)*u.percent)
    corona_map.plot(axes=ax_cor,clip_interval=(1,99.99)*u.percent)

    rosa_width = 1000 * rosa_dxy * u.arcsec
    zyla_width = 2048 * zyla_dxy * u.arcsec

    for i in range(len(list_of_rosa_bl)):
        rosa_bl_h = SkyCoord(
                list_of_rosa_bl[i][0]*u.arcsec,
                list_of_rosa_bl[i][1]*u.arcsec,
                frame=hmi_map.coordinate_frame)
        zyla_bl_h = SkyCoord(
                list_of_zyla_bl[i][0]*u.arcsec,
                list_of_zyla_bl[i][1]*u.arcsec,
                frame=hmi_map.coordinate_frame)

        hmi_map.draw_quadrangle(
                rosa_bl_h,
                axes=ax_hmi,
                width=rosa_width,
                height=rosa_width,
                edgecolor='C'+str(i+1),
                linewidth=2
        )
        hmi_map.draw_quadrangle(
                zyla_bl_h,
                axes=ax_hmi,
                width=zyla_width,
                height=zyla_width,
                edgecolor='C'+str(i+1),
                linewidth=2,
                linestyle='--'
        )

        rosa_bl_h = SkyCoord(
                list_of_rosa_bl[i][0]*u.arcsec,
                list_of_rosa_bl[i][1]*u.arcsec,
                frame=chrom_map.coordinate_frame)
        zyla_bl_h = SkyCoord(
                list_of_zyla_bl[i][0]*u.arcsec,
                list_of_zyla_bl[i][1]*u.arcsec,
                frame=chrom_map.coordinate_frame)

        chrom_map.draw_quadrangle(
                rosa_bl_h,
                axes=ax_chr,
                width=rosa_width,
                height=rosa_width,
                edgecolor='C'+str(i+1),
                linewidth=2,
                label=np.datetime64(pointings['TIME'][i],"m").astype(str)
        )
        chrom_map.draw_quadrangle(
                zyla_bl_h,
                axes=ax_chr,
                width=zyla_width,
                height=zyla_width,
                edgecolor='C'+str(i+1),
                linewidth=2,
                linestyle='--'
        )

        rosa_bl_h = SkyCoord(
                list_of_rosa_bl[i][0]*u.arcsec,
                list_of_rosa_bl[i][1]*u.arcsec,
                frame=corona_map.coordinate_frame)
        zyla_bl_h = SkyCoord(
                list_of_zyla_bl[i][0]*u.arcsec,
                list_of_zyla_bl[i][1]*u.arcsec,
                frame=corona_map.coordinate_frame)

        corona_map.draw_quadrangle(
                rosa_bl_h,
                axes=ax_cor,
                width=rosa_width,
                height=rosa_width,
                edgecolor='C'+str(i+1),
                linewidth=2
        )
        corona_map.draw_quadrangle(
                zyla_bl_h,
                axes=ax_cor,
                width=zyla_width,
                height=zyla_width,
                edgecolor='C'+str(i+1),
                linewidth=2,
                linestyle='--'
        )
    label_used = []
    for i in range(len(flare_xcd)):
        if flare_coordsys[i] == "degrees":
            unit = u.deg
        elif flare_coordsys[i] == 'arcsec':
            unit = u.arcsec

        if flare_goescls[i][0] == "B":
            pointc = "black"
            pointl = "B-Flare"
        elif flare_goescls[i][0] == "C":
            pointc = "C0"
            pointl = "C-Flare"
        elif flare_goescls[i][0] == "M":
            pointc = 'C2'
            pointl = 'M-Flare'
        elif flare_goescls[i][0] == "X":
            pointc = 'C3'
            pointl = "X-Flare"
        else:
            pointc = 'black'
            pointl = 'Flare'
        flare_coord = SkyCoord(
                flare_xcd[i]*unit,
                flare_ycd[i]*unit,
                obstime=flare_peak[i],
                observer='earth',
                frame=frames.HeliographicStonyhurst)
        ax_hmi.plot_coord(flare_coord,'X',color=pointc,markersize=7)
        if not any(pointl in label for label in label_used):
            ax_chr.plot_coord(flare_coord,
                     'X',
                     color=pointc,
                     markersize=7,
                     label=pointl)
            label_used.append(pointl)
        else:
            ax_chr.plot_coord(flare_coord,
                     'X',
                     color=pointc,
                     markersize=7)
        ax_cor.plot_coord(flare_coord,'X',color=pointc,markersize=7)

    for i in range(len(fe_xcd)):
        if fe_coordsys[i] == "degrees":
            unit = u.deg
            frame = frames.HeliographicStonyhurst
        elif fe_coordsys[i] == 'arcsec':
            unit = u.arcsec
            frame = frames.Helioprojective

        pointc = "C4"
        pointl = "Filament Eruption (+/-24 hours)"
        
        fe_coord = SkyCoord(
                fe_xcd[i] * unit,
                fe_ycd[i] * unit,
                obstime = t_start.astype(str),
                observer='earth',
                rsun=696000.0*u.km,
                distance=1*u.au,
                frame=frame)
        ax_hmi.plot_coord(fe_coord, "d", color=pointc, markersize=7)
        if i == 0:
            ax_chr.plot_coord(fe_coord,"d",color=pointc,label=pointl,markersize=7)
        else:
            ax_chr.plot_coord(fe_coord,"d",color=pointc,markersize=7)
        ax_cor.plot_coord(fe_coord,"d",color=pointc,markersize=7)

    for i in range(len(cr_xcd)):
        if cr_coordsys[i] == 'degrees':
            unit=u.deg
            frame=frames.HeliographicStonyhurst
        elif cr_coordsys[i] == 'arcsec':
            unit=u.arcsec
            frame=frames.Helioprojective

        pointc = "C5"
        pointl = "Coronal Rain (+/-24 hours)"

        cr_coord = SkyCoord(
                cr_xcd[i] * unit,
                cr_ycd[i] * unit,
                obstime = t_start.astype(str),
                observer='earth',
                rsun=696000.0*u.km,
                frame=frame,distance=1*u.au)
        ax_hmi.plot_coord(cr_coord, "o", color=pointc, markersize=7)
        if i == 0:
            ax_chr.plot_coord(cr_coord, "o", color=pointc, label=pointl, markersize=7)
        else:
            ax_chr.plot_coord(cr_coord, "o", color=pointc, markersize=7)
        ax_cor.plot_coord(cr_coord,"o",color=pointc,markersize=7)

    ax_chr.legend(loc='center',bbox_to_anchor=(0.5,-0.25),ncols=5)

    fig.text(0.1,0.9,"Dashed line = Zyla FOV",fontsize=14)
    fig.text(0.1,0.85,"Solid line = ROSA FOV",fontsize=14)

    fig.savefig(os.path.join(savedir,"context_image.jpg"),bbox_inches='tight')
    
