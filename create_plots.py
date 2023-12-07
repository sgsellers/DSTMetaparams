import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def create_seeing_plot(saveDir, seeingFile, referenceImages, instrument):
    """
    New general seeing plotting statements.
    :param saveDir: str
        Directory to save seeing plot
    :param seeingFile: str
        Path to seeing parameter file
    :param referenceImages: str
        Path to reference image file
    :param instrument: str
        Camera used. Allowed values are ['ROSA_GBAND', 'ROSA_4170', 'ROSA_CAK', 'ZYLA', 'IBIS']
    :return:
    """
    params = {
        'image.origin': 'lower',
        'image.interpolation': 'none',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 300,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'font.size': 24,
        'legend.fontsize': 12,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.figsize': [20, 20],
        'font.family': 'serif',
        'figure.facecolor': '#fef5e7'
    }

    plt.rcParams.update(params)
    seeing = np.load(seeingFile)
    refims = np.load(referenceImages)

    if "zyla" not in instrument.lower():
        timestamp_string = seeing['TIMESTAMPS'][0].astype('datetime64[s]').astype(str).replace(
            "T",
            "_"
        ).replace("-", "").replace(":", "")
        titletime_string = seeing['TIMESTAMPS'][0].astype('datetime64[s]').astype(str).replace("T", " ")
        if 'cak' in instrument.lower():
            instrtitle_string = 'ROSA Ca K'
        elif '4170' in instrument.lower():
            instrtitle_string = 'ROSA 4170'
        elif 'gband' in instrument.lower():
            instrtitle_string = 'ROSA G-Band'
        elif 'ibis' in instrument.lower():
            instrtitle_string = 'IBIS Broadband'
    else:
        timestamp_string = seeingFile.split("_zyla_seeing_quality.npy")[0].split("/")[-1]
        titletime_string = timestamp_string
        instrtitle_string = 'Zyla'

    figure_savestring = os.path.join(saveDir, timestamp_string + "_" + instrument.lower() + "_seeing_quicklook.png")

    fig = plt.figure(num=1, clear=True)
    gs = fig.add_gridspec(6, 12, hspace=0.5, wspace=1.5)
    ax_see = fig.add_subplot(gs[4, :5])
    ax_cpv = fig.add_subplot(gs[5, :5])
    ax_rms = fig.add_subplot(gs[4, 7:])
    ax_immn = fig.add_subplot(gs[5, 7:])

    ax_bad = fig.add_subplot(gs[:, :4])
    ax_med = fig.add_subplot(gs[:, 4:8])
    ax_good = fig.add_subplot(gs[:, 8:])

    if "zyla" in instrument.lower():
        primarySeeing = seeing['MFGS']
        secondarySeeing = None
        method = 'MFGS'
        xarray = np.arange(len(seeing['MFGS']))
    elif "rosa" in instrument.lower():
        primarySeeing = seeing['HSM']
        secondarySeeing = seeing['CENTRAL_HSM']
        method = 'H-S Mean'
        xarray = seeing['TIMESTAMPS']
    elif "ibis" in instrument.lower():
        primarySeeing = seeing['HSM']
        secondarySeeing = None
        method = 'H-S Mean'
        xarray = seeing['TIMESTAMPS']

    ax_see.scatter(
        xarray,
        primarySeeing,
        s=1,
        c='k',
        label='Full Image ' + method
    )
    if secondarySeeing is not None:
        ax_see.scatter(
            xarray,
            secondarySeeing,
            s=1,
            c='C1',
            label='Center Image' + method
        )
    ax_see.set_ylim(
        np.nanmean(primarySeeing) - 5*np.nanstd(primarySeeing),
        np.nanmean(primarySeeing) + 5*np.nanstd(primarySeeing)
    )
    ax_see.set_xlim(xarray[0], xarray[-1])
    ax_see.set_ylabel(method)
    ax_see.xaxis.set_ticklabels([])
    ax_see.legend(loc='lower left', scatterpoints=10)
    ax_see.set_title("Image Metrics")

    ax_cpv.scatter(
        xarray,
        seeing['CENTRAL_PIXEL'],
        s=1,
        c='k',
        label='Central Pixel Counts'
    )
    ax_cpv.scatter(
        xarray,
        seeing['CENTRAL_MEAN'],
        s=1,
        c='C1',
        label='Average of Central Subfield'
    )
    ax_cpv.set_xlim(xarray[0], xarray[-1])
    ax_cpv.set_ylabel("Central Variations")
    ax_cpv.legend(loc='lower left', scatterpoints=10)
    if ('rosa' in instrument.lower()) or ('ibis' in instrument.lower()):
        ax_cpv.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax_cpv.get_xticklabels():
            tick.set_rotation(30)

    ax_rms.scatter(
        xarray,
        seeing['IMAGE_RMS'],
        s=1,
        c='k',
        label='RMS Full'
    )
    if "ibis" not in instrument.lower():
        ax_rms.scatter(
            xarray,
            seeing['CENTRAL_RMS'],
            s=1,
            c='C1',
            label='Central RMS'
        )
    ax_rms.set_ylim(
        np.nanmean(seeing['IMAGE_RMS'] - 4*np.nanstd(seeing['IMAGE_RMS'])),
        np.nanmean(seeing['IMAGE_RMS'] + 4*np.nanstd(seeing['IMAGE_RMS']))
    )
    ax_rms.set_xlim(xarray[0], xarray[-1])
    ax_rms.xaxis.set_ticklabels([])
    ax_rms.set_ylabel("RMS")
    ax_rms.set_title("Image Metrics")
    ax_rms.legend(loc='lower left', scatterpoints=10)

    ax_immn.scatter(
        xarray,
        seeing['IMAGE_MEAN'],
        c='k',
        s=1,
        label='Image Mean'
    )
    ax_immn.set_ylim(
        np.nanmean(seeing['IMAGE_MEAN']) - 4*np.nanstd(seeing['IMAGE_MEAN']),
        np.nanmean(seeing['IMAGE_MEAN']) + 4*np.nanstd(seeing['IMAGE_MEAN'])
    )
    ax_immn.set_xlim(
        xarray[0], xarray[-1]
    )
    ax_immn.legend(loc='lower left', scatterpoints=10)
    if ('rosa' in instrument.lower()) or ('ibis' in instrument.lower()):
        ax_immn.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax_immn.get_xticklabels():
            tick.set_rotation(30)

    if 'zyla' in instrument.lower():
        bad_img = 'MIN_IMAGE'
        good_img = 'MAX_IMAGE'
    else:
        bad_img = 'MAX_IMAGE'
        good_img = 'MIN_IMAGE'

    if 'ibis' in instrument.lower():
        vmin_bad = (
                np.nanmean(refims[bad_img][200:800, 200:800]) -
                3*np.nanstd(refims[bad_img][200:800, 200:800])
        )
        vmax_bad = (
                np.nanmean(refims[bad_img][200:800, 200:800]) +
                3*np.nanstd(refims[bad_img][200:800, 200:800])
        )
        vmin_med = (
                np.nanmean(refims['MEDIAN_IMAGE'][200:800, 200:800]) -
                3 * np.nanstd(refims['MEDIAN_IMAGE'][200:800, 200:800])
        )
        vmax_med = (
                np.nanmean(refims['MEDIAN_IMAGE'][200:800, 200:800]) +
                3 * np.nanstd(refims['MEDIAN_IMAGE'][200:800, 200:800])
        )
        vmin_good = (
                np.nanmean(refims[good_img][200:800, 200:800]) -
                3 * np.nanstd(refims[good_img][200:800, 200:800])
        )
        vmax_good = (
                np.nanmean(refims[good_img][200:800, 200:800]) +
                3 * np.nanstd(refims[good_img][200:800, 200:800])
        )
    else:
        vmin_bad = (
                np.nanmean(refims[bad_img]) -
                3 * np.nanstd(refims[bad_img])
        )
        vmax_bad = (
                np.nanmean(refims[bad_img]) +
                3 * np.nanstd(refims[bad_img])
        )
        vmin_med = (
                np.nanmean(refims['MEDIAN_IMAGE']) -
                3 * np.nanstd(refims['MEDIAN_IMAGE'])
        )
        vmax_med = (
                np.nanmean(refims['MEDIAN_IMAGE']) +
                3 * np.nanstd(refims['MEDIAN_IMAGE'])
        )
        vmin_good = (
                np.nanmean(refims[good_img]) -
                3 * np.nanstd(refims[good_img])
        )
        vmax_good = (
                np.nanmean(refims[good_img]) +
                3 * np.nanstd(refims[good_img])
        )

    ax_bad.imshow(
        refims[bad_img],
        vmin=vmin_bad,
        vmax=vmax_bad
    )
    ax_bad.set_title("Quality 1$\\sigma$ Below Mean", pad=25)
    ax_bad.set_ylabel("Pixel Number")

    ax_med.imshow(
        refims['MEDIAN_IMAGE'],
        vmin=vmin_med,
        vmax=vmax_med
    )
    ax_med.set_title("Median Quality", pad=25)
    ax_med.set_xlabel("Pixel Number")

    ax_good.imshow(
        refims[good_img],
        vmin=vmin_good,
        vmax=vmax_good
    )
    ax_good.set_title("Quality 1$\\sigma$ Above Mean", pad=25)

    fig.suptitle(
        instrtitle_string + " Image Quality " + titletime_string,
        y=0.7
    )
    fig.savefig(figure_savestring, bbox_inches='tight')
    plt.close(fig)
    plt.cla()
    return


def create_timing_plot(saveDir):
    """
    Generates plot of instrument obs timings.
    Calls various obssum functions, and builds bar chart from there.

    :param saveDir:
    :return:
    """