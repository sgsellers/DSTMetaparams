import numpy as np
import sean_tools as st
import zyla_tools as zt
import glob
import gc
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
import os
from tqdm import tqdm
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits


def automatic(indir, zyla_shape=None):
    """Master function to determine seeing parameters for ROSA and Zyla"""
    if indir[-1] != "/":
        indir += "/"
    rosa_seeing(indir)
    zyla_seeing(indir, zyla_shape=zyla_shape)
    ibis_seeing(indir)
    #rosa_seeing(indir)
    create_rosa_plots(indir)
    create_zyla_plots(indir)
    create_ibis_plots(indir)

@dask.delayed
def _hsm_and_params(fname):
    file = fits.open(fname)
    exts = len(file) - 1
    timestamps = []
    hsm = np.zeros(exts)
    hsm_central = np.zeros(exts)
    central_pix = np.zeros(exts)
    central_subfield = np.zeros(exts)
    frame_mean = np.zeros(exts)
    frame_rms = np.zeros(exts)
    central_rms = np.zeros(exts)
    for i in range(exts):
        array = file[i+1].data / float(file[0].header['EXPOSURE'][:-4])
        try:
            timestamps.append(file[i+1].header['DATE'])
            hsm[i] = st.hs_mean(array)
            hsm_central[i] = st.hs_mean(array[450:550, 450:550])
            central_pix[i] = array[501,502]
            central_subfield[i] = np.nanmean(array[450:550, 450:550])
            frame_mean[i] = np.nanmean(array)
            frame_rms[i] = np.sum((array - frame_mean[i])**2)/(1002*1004)
            central_rms[i] = np.sum((array[450:550, 450:550] - central_subfield[i])**2)/(100**2)
        except:
            timestamps.append(np.nan)
            hsm[i] = np.nan
            hsm_central[i] = np.nan
            central_pix[i] = np.nan
            central_subfield[i] = np.nan
            frame_mean[i] = np.nan
            frame_rms[i] = np.nan
            central_rms[i] = np.nan
    return [exts, timestamps, hsm, hsm_central, central_pix, central_subfield, frame_mean, frame_rms, central_rms]

@dask.delayed
def _ibis_hsm(fname):
    try:
        file = fits.open(fname)
    except:
        return [0, [], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
    exts = len(file) -1
    timestamps = []
    hsm = np.zeros(exts)
    central_pix = np.zeros(exts)
    central_subfield = np.zeros(exts)
    frame_mean = np.zeros(exts)
    frame_rms = np.zeros(exts)
    for i in range(exts):
        try:
            array = file[i+1].data[200:800,200:800] / float(file[i+1].header['EXPTIME'])
            timestamps.append(file[i+1].header['DATE-OBS'])
            hsm[i] = st.hs_mean(array)
            central_pix[i] = array[300,300]
            central_subfield[i] = np.nanmean(array[250:350,250:350])
            frame_mean[i] = np.nanmean(array)
            frame_rms[i] = np.sum((array - frame_mean[i])**2)/(600*600)
        except:
            timestamps.append(np.nan)
            hsm[i] = np.nan
            central_pix[i] = np.nan
            central_subfield = np.nan
            frame_mean[i] = np.nan
            frame_rms[i] = np.nan
    return [exts, timestamps, hsm, central_pix, central_subfield, frame_mean, frame_rms]

@dask.delayed
def _mfgs_and_params(fname, zyla_shape=(2048,2048)):
    try:
        if "fits" in fname:
            array = fits.open(fname)[1].data
        else:
            array = zt.read_zyla(fname, dataShape=zyla_shape, imageShape=zyla_shape)
        mfgs = st.mfgs(array)
        central_pixel = array[int(array.shape[0]/2),int(array.shape[1]/2)]
        central_subfield = np.nanmean(
                array[
                    int(array.shape[0]/2 - 100):int(array.shape[0]/2 + 100),
                    int(array.shape[1]/2 - 100):int(array.shape[1]/2 + 100)
                ]
        )
        frame_mean = np.nanmean(array)
        frame_rms = np.nansum(
                (array - frame_mean)**2)/(array.shape[0] * array.shape[1])
        central_rms = np.nansum(
                (array[
                    int(array.shape[0]/2 - 100):int(array.shape[0]/2 + 100),
                    int(array.shape[1]/2 - 100):int(array.shape[1]/2 + 100)
                ] - central_subfield)**2)/(200**2)
    except:
        mfgs = np.nan
        central_pixel = np.nan
        central_subfield = np.nan
        frame_mean = np.nan
        frame_rms = np.nan
        central_rms = np.nan
    return [fname, np.array([mfgs, central_pixel, central_subfield, frame_mean, frame_rms, central_rms])]


def _find_timestamp_gaps(timestamp_array, max_gap=np.timedelta64(5, 'm'), min_entries=100):
    dts = np.diff(timestamp_array)
    gap_edges = np.pad(np.where(dts >= max_gap)[0], (1, 1))
    long_windows = np.where(np.diff(gap_edges) >= min_entries)[0]
    window_indices = np.zeros((2, len(long_windows)))
    for i in range(len(long_windows)):
        window_indices[:, i] = [gap_edges[long_windows[i]], gap_edges[long_windows[i] + 1]]
    if len(gap_edges) == 2:
        window_indices = np.zeros((2, 1))
        window_indices[1, 0] = len(timestamp_array)
    return window_indices

def rosa_seeing(indir):
    """Searches for ROSA data files and determines seeing parameters for them."""
    rosa_filters = ['4170', 'gband','cak']
    for i in rosa_filters:
        if any(i+'_seeing_quality' in fl for fl in glob.glob(indir + '*')):
            return
        working_directory = glob.glob(
                indir + "**/level0/**/*"+i+"*/",
                recursive=True)
        if len(working_directory) > 0:
            working_directory = working_directory[0]
            params = []
            # I started writing this, and realized; why write three list comprehensions back to back when I could write
            # one big, unreadable one. So I did.
            # flist = sorted(glob.glob(working_directory + "*fit*"))
            # flist = [x for x in flist if "dark" not in x and "flat" not in x]
            # unique_flist = set(["_".join(x.split("_")[:-1]) for x in flist_noflatdark])

            # This is the worst thing I've ever written. It's bad *on purpose*. As a bit.
            # If you're not me, just know that it searches for fit files, then cuts out the flats, darks,
            # and files with too few repetitions (10 is my current fudge)
            # I'm not sorry. This is what you get for looking at my code.
            flist = [item for sublist in [x for x in [glob.glob(y + "*fit*") for y in set(["_".join(z.split("_")[:-1]) for z in [q for q in sorted(glob.glob(working_directory + "*fit*")) if "dark" not in q and "flat" not in q]]) if len(glob.glob(y + "*fit*")) > 10]] for item in sublist]
            # Note to Future Sean:
            # Hey Idiot.
            # Parallel is definitely faster.
            # Quit FUCKING switching between parallel and serial.
            # This is 2-3 times as fast as the serial version.
            # Could it be faster?
            # Probably
            # I don't know.
            # I'm you. 
            # As established, an idiot.
            # You might be able to do it if you parallelized in-file, at the extension level.
            # But for right now? Quit messing with it.
            # You're only making it worse.
            # And I hate you.
            # <3
            for j in range(len(flist)):
                res = _hsm_and_params(flist[j])
                params.append(res)
            print("Computing Seeing Params for ROSA", i)
            with ProgressBar():
                results = dask.compute(params)[0]

            exts = [x[0] for x in results]
            t_arr = np.concatenate([x[1] for x in results])
            hsm = np.concatenate([x[2] for x in results])
            hsm_central = np.concatenate([x[3] for x in results]).flatten()
            central_pix = np.concatenate([x[4] for x in results]).flatten()
            central_subfield = np.concatenate([x[5] for x in results])
            frame_mean = np.concatenate([y for y in [x[6] for x in results]])
            frame_rms = np.concatenate([y for y in [x[7] for x in results]])
            central_rms = np.concatenate([y for y in [x[8] for x in results]])
               
            t_nancut = t_arr != 'nan'

            timestamp = t_arr[t_nancut].astype('datetime64')
            tsort = np.argsort(timestamp)
            timestamp = timestamp[tsort]
            hsm = hsm[t_nancut][tsort]
            hsm_central = hsm_central[t_nancut][tsort]
            central_pixel = central_pix[t_nancut][tsort]
            central_mean = central_subfield[t_nancut][tsort]
            frame_mean = frame_mean[t_nancut][tsort]
            frame_rms = frame_rms[t_nancut][tsort]
            central_rms = central_rms[t_nancut][tsort]
            idxs = _find_timestamp_gaps(timestamp).astype(int)
            seeing_params = []
            images = []
            print(idxs)
            for k in range(idxs.shape[1]):
                lo = idxs[0, k]
                hi = idxs[1, k]

                ts_pointing = timestamp[lo:hi]
                hsm_pointing = hsm[lo:hi]
                hsm_central_pointing = hsm_central[lo:hi]
                central_pixel_pointing = central_pixel[lo:hi]
                central_mean_pointing = central_mean[lo:hi]
                frame_mean_pointing = frame_mean[lo:hi]
                frame_rms_pointing = frame_rms[lo:hi]
                central_rms_pointing = central_rms[lo:hi]

                # Pull median, and +/-1 sigma images

                median_index = st.find_nearest(np.nan_to_num(hsm_pointing), np.nanmedian(hsm_pointing)) + lo
                ctr = 0
                img_idx = 0
                while ctr < median_index:
                    ctr += exts[img_idx]
                    img_idx += 1
                median_image = fits.open(flist[img_idx-1])[median_index - (ctr - exts[img_idx-1])].data
                ####
                min_index = st.find_nearest(
                        np.nan_to_num(hsm_pointing), 
                        np.nanmean(hsm_pointing) - np.nanstd(hsm_pointing)
                ) + lo

                ctr = 0
                img_idx = 0
                while ctr < min_index:
                    ctr += exts[img_idx]
                    img_idx += 1
                min_image = fits.open(flist[img_idx-1])[min_index - (ctr - exts[img_idx-1])].data
                ####
                max_index = st.find_nearest(
                        np.nan_to_num(hsm_pointing), 
                        np.nanmean(hsm_pointing) + np.nanstd(hsm_pointing)
                ) + lo

                ctr = 0
                img_idx = 0
                while ctr < max_index:
                    ctr += exts[img_idx]
                    img_idx += 1
                max_image = fits.open(flist[img_idx-1])[max_index - (ctr - exts[img_idx-1])].data

                seeing_info = np.rec.fromarrays(
                        [ts_pointing, 
                         hsm_pointing, 
                         hsm_central_pointing,
                         central_pixel_pointing,
                         central_mean_pointing,
                         frame_mean_pointing,
                         frame_rms_pointing,
                         central_rms_pointing],
                        names=['TIMESTAMPS',
                               'HSM',
                               'CENTRAL_HSM',
                               'CENTRAL_PIXEL',
                               'CENTRAL_MEAN',
                               'IMAGE_MEAN',
                               'IMAGE_RMS',
                               'CENTRAL_RMS']
                )
                key_images = np.rec.fromarrays(
                        [median_image, min_image, max_image],
                        names=['MEDIAN_IMAGE', 'MIN_IMAGE', 'MAX_IMAGE']
                )
                good_str = True
                zctr=0
                while good_str:
                    good_str = np.isnan(ts_pointing[zctr])
                    zctr +=1
                h = (ts_pointing[zctr].astype("datetime64[h]") - ts_pointing[0].astype("datetime64[D]")).astype(int).astype(str).zfill(2)
                m = (ts_pointing[zctr].astype("datetime64[m]") - ts_pointing[0].astype("datetime64[h]")).astype(int).astype(str).zfill(2)
                s = (ts_pointing[zctr].astype("datetime64[s]") - ts_pointing[0].astype("datetime64[m]")).astype(int).astype(str).zfill(2)

                seeing_params.append(seeing_info)
                images.append(key_images)
                np.save(os.path.join(indir, h+m+s + "_" + i + "_seeing_quality.npy"), seeing_info)
                np.save(os.path.join(indir, h+m+s + "_" + i + "_key_images.npy"), key_images)

def ibis_seeing(indir):
    """For IBIS, we'll use the cotemporal whitelight channel to get seeing information."""
    if any('ibis_seeing_quality' in fl for fl in glob.glob(indir + '*')):
        return
    whitelight_dirs = glob.glob(
            indir + "**/level0/**/*whitelight*/ScienceObservation/*/", recursive=True
    )
    for i in range(len(whitelight_dirs)):
        flist = sorted(glob.glob(whitelight_dirs[i] + "s*.fits"))
        params = []
        for j in range(len(flist)):
            res = _ibis_hsm(flist[j])
            params.append(res)
        print("Computing Seeing Params for IBIS whitelight")
        with ProgressBar():
            results = dask.compute(params)[0]

        exts = [x[0] for x in results]
        t_arr = np.concatenate([x[1] for x in results])
        hsm = np.concatenate([x[2] for x in results])
        central_pix = np.concatenate([x[3] for x in results]).flatten()
        central_subfield = np.concatenate([x[4] for x in results])
        frame_mean = np.concatenate([y for y in [x[5] for x in results]])
        frame_rms = np.concatenate([y for y in [x[6] for x in results]])

        t_nancut = t_arr != 'nan'

        timestamp = t_arr[t_nancut].astype('datetime64')
        tsort = np.argsort(timestamp)
        timestamp = timestamp[tsort]
        hsm = hsm[t_nancut][tsort]
        central_pixel = central_pix[t_nancut][tsort]
        central_mean = central_subfield[t_nancut][tsort]
        frame_mean = frame_mean[t_nancut][tsort]
        frame_rms = frame_rms[t_nancut][tsort]
        seeing_params = []
        images = []
        # Pull median, and +/-1 sigma images

        median_index = st.find_nearest(np.nan_to_num(hsm), np.nanmedian(hsm))
        ctr = 0
        img_idx = 0
        while ctr < median_index:
            ctr += exts[img_idx]
            img_idx += 1
        median_image = fits.open(flist[img_idx-1])[median_index - (ctr - exts[img_idx-1])].data
        ####
        min_index = st.find_nearest(
                np.nan_to_num(hsm), 
                np.nanmean(hsm) - np.nanstd(hsm)
        )

        ctr = 0
        img_idx = 0
        while ctr < min_index:
            ctr += exts[img_idx]
            img_idx += 1
        min_image = fits.open(flist[img_idx-1])[min_index - (ctr - exts[img_idx-1])].data
        ####
        max_index = st.find_nearest(
                np.nan_to_num(hsm), 
                np.nanmean(hsm) + np.nanstd(hsm)
        )

        ctr = 0
        img_idx = 0
        while ctr < max_index:
            ctr += exts[img_idx]
            img_idx += 1
        max_image = fits.open(flist[img_idx-1])[max_index - (ctr - exts[img_idx-1])].data

        seeing_info = np.rec.fromarrays(
                [timestamp, 
                 hsm, 
                 central_pixel,
                 central_mean,
                 frame_mean,
                 frame_rms],
                names=['TIMESTAMPS',
                       'HSM',
                       'CENTRAL_PIXEL',
                       'CENTRAL_MEAN',
                       'IMAGE_MEAN',
                       'IMAGE_RMS']
        )
        key_images = np.rec.fromarrays(
                [median_image, min_image, max_image],
                names=['MEDIAN_IMAGE', 'MIN_IMAGE', 'MAX_IMAGE']
        )

        timestamp = whitelight_dirs[i].split("/")[-2].split("_")[-1]
        np.save(os.path.join(indir, timestamp + "_ibis_seeing_quality.npy"), seeing_info)
        np.save(os.path.join(indir, timestamp + "_ibis_key_images.npy"), key_images)

def create_ibis_plots(indir):
    params = {
            'image.origin':'lower',
            'image.interpolation':'none',
            'image.cmap':'gray',
            'axes.grid':False,
            'savefig.dpi':300,
            'axes.labelsize':24,
            'axes.titlesize':24,
            'font.size':24,
            'legend.fontsize':12,
            'xtick.labelsize':20,
            'ytick.labelsize':20,
            'figure.figsize':[20,20],
            'font.family':'serif',
            'figure.facecolor':'#fef5e7'
    }

    plt.rcParams.update(params)

    if any("ibis_seeing_quicklook" in fl for fl in glob.glob(indir + "*")):
        return
    pointings = sorted(glob.glob(indir + "*ibis_seeing_quality.npy"))
    refs = sorted(glob.glob(indir + "*ibis_key_images.npy"))
    for i in range(len(pointings)):
        seeing = np.load(pointings[i])
        images = np.load(refs[i])
        timestamps = seeing['TIMESTAMPS']
        hsm = seeing['HSM']
        single_pixel_variations = seeing['CENTRAL_PIXEL']
        central_pixel_variations = seeing['CENTRAL_MEAN']
        root_mean_square = seeing["IMAGE_RMS"]
        image_mean = seeing["IMAGE_MEAN"]
        bad_frame = images["MAX_IMAGE"]
        good_frame = images["MIN_IMAGE"]
        okay_frame = images["MEDIAN_IMAGE"]

        h = (timestamps[0].astype("datetime64[h]") - timestamps[0].astype("datetime64[D]")).astype(int).astype(str).zfill(2)
        m = (timestamps[0].astype("datetime64[m]") - timestamps[0].astype("datetime64[h]")).astype(int).astype(str).zfill(2)
        s = (timestamps[0].astype("datetime64[s]") - timestamps[0].astype("datetime64[m]")).astype(int).astype(str).zfill(2)

        savestr = os.path.join(indir, "ibis_seeing_quicklook_" + h + m + s + ".png")

        fig = plt.figure()
        gs = fig.add_gridspec(6,12,hspace = 0.5,wspace = 5)

        ax_hsm = fig.add_subplot(gs[4,:5])
        ax_hsm.scatter(
                timestamps,
                hsm,
                s=1,
                c='k',
                label = 'Full Image HSM'
        )

        ax_hsm.set_ylim(
                np.nanmean(hsm) - 3*np.nanstd(hsm),
                np.nanmean(hsm) + 3*np.nanstd(hsm))
        ax_hsm.set_xlim(timestamps[0],timestamps[-1])
        ax_hsm.set_ylabel("H-S Mean")
        ax_hsm.set_xticks([])
        ax_hsm.legend(loc = 'lower left',scatterpoints = 10)
        ax_hsm.set_title("Image Metrics")
        
        ####

        ax_cpv = fig.add_subplot(gs[5,:5])

        ax_cpv.scatter(
                timestamps,
                single_pixel_variations,
                s=1,
                c='k',
                label = 'Central Pixel Counts'
        )

        ax_cpv.scatter(
                timestamps,
                central_pixel_variations,
                s=1,
                c='C1',
                label = 'Central Subfield Counts'
        )

        ax_cpv.set_ylim(
                np.nanmean(single_pixel_variations) - 3*np.nanstd(single_pixel_variations),
                np.nanmean(single_pixel_variations) + 3*np.nanstd(single_pixel_variations)
        )

        ax_cpv.set_xlim(
                timestamps[0],
                timestamps[-1]
        )
        ax_cpv.set_ylabel("Central Variations")
        ax_cpv.legend(
                loc = 'lower left',
                scatterpoints = 10
        )
        ax_cpv.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax_cpv.get_xticklabels():
            tick.set_rotation(30)
        
        ####

        ax_rms = fig.add_subplot(gs[4,7:])
        ax_rms.scatter(
                timestamps,
                root_mean_square,
                s=1,
                c='k',
                label = 'RMS Full'
        )

        ax_rms.set_ylim(
                np.nanmean(root_mean_square) - 3*np.nanstd(root_mean_square),
                np.nanmean(root_mean_square) + 3*np.nanstd(root_mean_square)
        )

        ax_rms.set_xlim(timestamps[0],timestamps[-1])
        ax_rms.set_ylabel("RMS")
        ax_rms.set_title("Image Metrics")
        ax_rms.set_xticks([])
        ax_rms.legend(loc = 'lower left',scatterpoints = 10)

        ####

        ax_immn = fig.add_subplot(gs[5,7:])
        ax_immn.scatter(
                timestamps,
                image_mean,
                s=1,
                c='k',
                label = 'Image Mean'
        )

        ax_immn.set_ylim(
                np.nanmean(image_mean) - 3*np.nanstd(image_mean),
                np.nanmean(image_mean) + 3*np.nanstd(image_mean)
        )
        ax_immn.set_xlim(timestamps[0],timestamps[-1])
        ax_immn.legend(loc = 'lower left',scatterpoints = 10)
        ax_immn.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax_immn.get_xticklabels():
            tick.set_rotation(30)

        ####

        ax_bad = fig.add_subplot(gs[:,:4])
        ax_bad.imshow(bad_frame[200:800,200:800])
        ax_bad.set_title("Quality 1sigma worse than Mean",pad=25)
        ax_bad.set_ylabel("Pixel Number")

        ax_med = fig.add_subplot(gs[:,4:8])
        ax_med.imshow(okay_frame[200:800,200:800])
        ax_med.set_title("Quality of Median",pad=25)
        ax_med.set_xlabel("Pixel Number")

        ax_good = fig.add_subplot(gs[:,8:])
        ax_good.imshow(good_frame[200:800,200:800])
        ax_good.set_title("Quality 1sigma better than Mean",pad=25)
        
        fig.suptitle("IBIS Image Quality " + timestamps[0].astype(str),y = 0.7)
        fig.savefig(savestr,bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close('all')

def zyla_seeing(indir, zyla_shape=None):
    """We might be able to parallelize this one. We may HAVE to. MFGS is a lot slower than HSM.
    Unfortunately, HSM isn't too sensitive to chromospheric structures. Neither is MFGS. But it's better."""
    if any('zyla_seeing_quality' in fl for fl in glob.glob(indir + '*')):
        return
    
    zyla_dir = glob.glob(
            indir + "**/level0/**/*Zyla*/", recursive=True
    ) + glob.glob(
            indir + "**/level0/**/*zyla*/", recursive=True
    )

    if len(zyla_dir) > 0:
        zyla_dir = zyla_dir[0]
        # First we gotta do the dumb search for which sub-directory is our obs directory.
        # It's named after how the observers felt that day. These are the most common labels.
        # It is unfortunately not comprehensive.

        obs_dir = glob.glob(zyla_dir + "**/DBJ*obs*/",
                            recursive=True)
        obs_dir2 = glob.glob(zyla_dir + "**/DBJ*obs*/",
                             recursive=True)
        fil_dir = glob.glob(zyla_dir + "**/DBJ*fil*/",
                            recursive=True)
        limb_dir = glob.glob(zyla_dir + "**/DBJ*limb*/",
                             recursive=True)
        ar_dir = glob.glob(zyla_dir + "**/DBJ*_ar*/",
                           recursive=True)
        ar_dir2 = glob.glob(zyla_dir + "**/DBJ*_AR*/",
                            recursive=True)
        bur_dir = glob.glob(zyla_dir + "**/DBJ*burst*/",
                            recursive=True)
        data_dirs = list(set(
            obs_dir+
            obs_dir2+
            fil_dir+
            limb_dir+
            ar_dir+
            ar_dir2+
            bur_dir))
        if len(data_dirs) == 0:
            all_dirs = glob.glob(zyla_dir + "**/DBJ*/", recursive=True)
            data_dirs = []
            cal_tags = ['flat','dark','dot','pin','ph','error','line','targ']
            for i in range(len(all_dirs)):
                if not any(x in all_dirs[i] for x in cal_tags):
                    data_dirs.append(all_dirs[i])

        for i in data_dirs:
            end_folder = i.split("/")[-2]

            folder_tags = end_folder.split("_")[2:]
            timetag = end_folder
            for j in range(len(folder_tags)):
                if len(folder_tags[j]) >= 4:
                    if folder_tags[j].isnumeric():
                        timetag = folder_tags[j]
                    elif folder_tags[j][:4].isnumeric():
                        timetag = folder_tags[j]
            flist = zt.order_zyla_filelist(glob.glob(i + '*.dat*'))
            # Now we're going to use the dask delayed mfgs function above.
            # Since dask is lazily computed (i.e., not evaluated until explicitly called)
            # Our loop can be quite simple...
            param_list = []
            for k in range(len(flist)):
                param_list.append(_mfgs_and_params(flist[k], zyla_shape=zyla_shape))
            print("Computing Seeing Params for Zyla", i)
            with ProgressBar():
                params = dask.compute(param_list)[0]
            fnames = [x[0] for x in params]
            fname_sort = zt.argsort_zyla_filelist(fnames)
            fnames = np.array(fnames)[fname_sort]
            mfgs = np.array([x[1][0] for x in params])[fname_sort]
            central_pix = np.array([x[1][1] for x in params])[fname_sort]
            central_subfield = np.array([x[1][2] for x in params])[fname_sort]
            frame_mean = np.array([x[1][3] for x in params])[fname_sort]
            frame_rms = np.array([x[1][4] for x in params])[fname_sort]
            central_rms = np.array([x[1][5] for x in params])[fname_sort]


            seeing_info = np.rec.fromarrays(
                    [
                        fnames,
                        mfgs,
                        central_pix,
                        central_subfield,
                        frame_mean,
                        frame_rms,
                        central_rms
                    ],
                    names = [
                        'FILENAME',
                        'MFGS',
                        'CENTRAL_PIXEL',
                        'CENTRAL_MEAN',
                        'IMAGE_MEAN',
                        'IMAGE_RMS',
                        'CENTRAL_RMS'
                    ]
            )
            
            savestr = os.path.join(indir, timetag + "_zyla_seeing_quality.npy")
            np.save(savestr, seeing_info)

            med_idx = st.find_nearest(np.nan_to_num(mfgs), np.nanmedian(mfgs))
            min_idx = st.find_nearest(np.nan_to_num(mfgs), np.nanmean(mfgs) - np.nanstd(mfgs))
            max_idx = st.find_nearest(np.nan_to_num(mfgs), np.nanmean(mfgs) + np.nanstd(mfgs))
            print("Median,",flist[med_idx])
            print("Min,",flist[min_idx])
            print("Max,",flist[max_idx])
            if "fits" in flist[med_idx]:
                median_image = fits.open(flist[med_idx])[1].data
                min_image = fits.open(flist[min_idx])[1].data
                max_image = fits.open(flist[max_idx])[1].data
            else:
                median_image = zt.read_zyla(flist[med_idx], dataShape=zyla_shape, imageShape=zyla_shape)
                min_image = zt.read_zyla(flist[min_idx], dataShape=zyla_shape, imageShape=zyla_shape)
                max_image = zt.read_zyla(flist[max_idx], dataShape=zyla_shape, imageShape=zyla_shape)

            key_images = np.rec.fromarrays(
                    [median_image, min_image, max_image],
                    names=['MEDIAN_IMAGE','MIN_IMAGE','MAX_IMAGE'])

            savestr = os.path.join(indir, timetag + "_zyla_key_images.npy")
            np.save(savestr, key_images)

def create_zyla_plots(indir):
    params = {
            'image.origin':'lower',
            'image.interpolation':'none',
            'image.cmap':'gray',
            'axes.grid':False,
            'savefig.dpi':300,
            'axes.labelsize':24,
            'axes.titlesize':24,
            'font.size':24,
            'legend.fontsize':12,
            'xtick.labelsize':20,
            'ytick.labelsize':20,
            'figure.figsize':[20,20],
            'font.family':'serif',
            'figure.facecolor':'#fef5e7'
    }

    plt.rcParams.update(params)

    if any("zyla_seeing_quicklook.png" in fl for fl in glob.glob(indir + "*")):
        return
    pointings = sorted(glob.glob(indir + "*zyla_seeing_quality.npy"))
    refs = sorted(glob.glob(indir + "*zyla_key_images.npy"))
    for i in range(len(pointings)):
        seeing = np.load(pointings[i])
        images = np.load(refs[i])

        mfgs = seeing['MFGS']
        central_pixel = seeing['CENTRAL_PIXEL']
        central_subfield = seeing['CENTRAL_MEAN']
        image_mean = seeing['IMAGE_MEAN']
        image_rms = seeing['IMAGE_RMS']
        central_rms = seeing['CENTRAL_RMS']

        bad_image = images['MIN_IMAGE']
        good_image = images['MAX_IMAGE']
        median_image = images['MEDIAN_IMAGE']

        timestr = pointings[i].split("_zyla_seeing_quality.npy")[0].split("/")[-1]
        savestr = pointings[i].split("_zyla_seeing_quality.npy")[0] + "_zyla_seeing_quicklook.png"

        fig = plt.figure()
        gs = fig.add_gridspec(6,12,hspace=0.5,wspace=5)

        xarr = np.arange(len(mfgs))

        ax_mfgs = fig.add_subplot(gs[4,:5])
        ax_mfgs.scatter(
                xarr,
                mfgs,
                s=1,
                c='k',
                label='MFGS'
        )

        ax_mfgs.set_ylim(
                np.nanmean(mfgs) - 5*np.nanstd(mfgs),
                np.nanmean(mfgs) + 5*np.nanstd(mfgs))
        ax_mfgs.set_xlim(xarr[0],xarr[-1])
        ax_mfgs.set_ylabel("Zyla MFGS")
        ax_mfgs.set_xticks([])
        ax_mfgs.set_title("Image Metrics")

        ####

        ax_cpv = fig.add_subplot(gs[5,:5])
        
        ax_cpv.scatter(
                xarr,
                central_pixel,
                s=1,
                c='k',
                label='Central Pixel Counts'
        )

        ax_cpv.scatter(
                xarr,
                central_subfield,
                s=1,
                c='C1',
                label='Average of Central Subfield'
        )

        ax_cpv.set_xlim(xarr[0],xarr[-1])
        ax_cpv.set_ylabel("Central Variations")
        ax_cpv.legend(
                loc='lower left',
                scatterpoints=10)

        ####

        ax_rms = fig.add_subplot(gs[4,7:])
        ax_rms.scatter(
                xarr,
                image_rms,
                s=1,
                c='k',
                label='RMS Image'
        )
        
        ax_rms.scatter(
                xarr,
                central_rms,
                s=1,
                c='C1',
                label='RMS Center'
        )

        ax_rms.set_ylim(
                np.nanmean(image_rms) - 5*np.nanstd(image_rms),
                np.nanmean(image_rms) + 5*np.nanstd(image_rms))

        ax_rms.set_xlim(xarr[0],xarr[-1])
        ax_rms.set_ylabel("RMS")
        ax_rms.set_title("Image Metrics")
        ax_rms.set_xticks([])
        ax_rms.legend(loc='lower left',scatterpoints=10)

        ####

        ax_immn = fig.add_subplot(gs[5,7:])
        ax_immn.scatter(
                xarr,
                image_mean,
                s=1,
                c='k',
                label='Image Mean'
        )

        ax_immn.set_ylim(
                np.nanmean(image_mean) - 3*np.nanstd(image_mean),
                np.nanmean(image_mean) + 3*np.nanstd(image_mean))

        ax_immn.set_xlim(xarr[0],xarr[-1])
        ax_immn.legend(loc='lower left',scatterpoints=10)
        
        ####

        ax_bad = fig.add_subplot(gs[:,:4])
        ax_bad.imshow(
                bad_image,
                vmin=np.nanmean(bad_image)-3*np.nanstd(bad_image),
                vmax=np.nanmean(bad_image)+3*np.nanstd(bad_image))
        ax_bad.set_title("Quality 1sigma worse than Mean",pad=25)
        ax_bad.set_ylabel("Pixel Number")

        ax_med = fig.add_subplot(gs[:,4:8])
        ax_med.imshow(median_image,
                vmin=np.nanmean(median_image)-3*np.nanstd(median_image),
                vmax=np.nanmean(median_image)+3*np.nanstd(median_image))
        ax_med.set_title("Quality of Median",pad=25)
        ax_med.set_xlabel("Pixel Number")

        ax_good = fig.add_subplot(gs[:,8:])
        ax_good.imshow(good_image,
                vmin=np.nanmean(good_image)-3*np.nanstd(good_image),
                vmax=np.nanmean(good_image)+3*np.nanstd(good_image))
        ax_good.set_title("Quality 1sigma better than Mean",pad=25)

        fig.suptitle(timestr + " Zyla Image Quality",y=0.7)
        fig.savefig(savestr,bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close('all')

def create_rosa_plots(indir):
    params = {
            'image.origin':'lower',
            'image.interpolation':'none',
            'image.cmap':'gray',
            'axes.grid':False,
            'savefig.dpi':300,
            'axes.labelsize':24,
            'axes.titlesize':24,
            'font.size':24,
            'legend.fontsize':12,
            'xtick.labelsize':20,
            'ytick.labelsize':20,
            'figure.figsize':[20,20],
            'font.family':'serif',
            'figure.facecolor':'#fef5e7'
    }

    plt.rcParams.update(params)

    rosa_filter = ['cak','3500','4170','gband']
    for j in rosa_filter:
        if any(j + "_seeing_quicklook" in fl for fl in glob.glob(indir + "*")):
            return
        pointings = sorted(glob.glob(indir + "*"+j+"_seeing_quality.npy"))
        refs = sorted(glob.glob(indir + "*"+j+"_key_images.npy"))
        for i in range(len(pointings)):
            seeing = np.load(pointings[i])
            images = np.load(refs[i])
            timestamps = seeing['TIMESTAMPS']
            hsm = seeing['HSM']
            central_hsm = seeing['CENTRAL_HSM']
            single_pixel_variations = seeing['CENTRAL_PIXEL']
            central_pixel_variations = seeing['CENTRAL_MEAN']
            root_mean_square = seeing["IMAGE_RMS"]
            central_root_mean_square = seeing["CENTRAL_RMS"]
            image_mean = seeing["IMAGE_MEAN"]
            bad_frame = images["MAX_IMAGE"]
            good_frame = images["MIN_IMAGE"]
            okay_frame = images["MEDIAN_IMAGE"]

            h = (timestamps[0].astype("datetime64[h]") - timestamps[0].astype("datetime64[D]")).astype(int).astype(str).zfill(2)
            m = (timestamps[0].astype("datetime64[m]") - timestamps[0].astype("datetime64[h]")).astype(int).astype(str).zfill(2)
            s = (timestamps[0].astype("datetime64[s]") - timestamps[0].astype("datetime64[m]")).astype(int).astype(str).zfill(2)

            savestr = os.path.join(indir, j + "_seeing_quicklook_" + h + m + s + ".png")
            filter_name = j

            fig = plt.figure()
            gs = fig.add_gridspec(6,12,hspace = 0.5,wspace = 5)

            ax_hsm = fig.add_subplot(gs[4,:5])
            ax_hsm.scatter(
                    timestamps,
                    hsm,
                    s=1,
                    c='k',
                    label = 'Full Image HSM'
            )

            ax_hsm.scatter(
                    timestamps,
                    central_hsm,
                    s=1,
                    c='C1',
                    label = 'Center Image HSM'
            )


            ax_hsm.set_ylim(
                    np.nanmean(hsm) - 3*np.nanstd(hsm),
                    np.nanmean(hsm) + 3*np.nanstd(hsm))
            ax_hsm.set_xlim(timestamps[0],timestamps[-1])
            ax_hsm.set_ylabel("H-S Mean")
            ax_hsm.set_xticks([])
            ax_hsm.legend(loc = 'lower left',scatterpoints = 10)
            ax_hsm.set_title("Image Metrics")
            
            ####

            ax_cpv = fig.add_subplot(gs[5,:5])

            ax_cpv.scatter(
                    timestamps,
                    single_pixel_variations,
                    s=1,
                    c='k',
                    label = 'Central Pixel Counts'
            )

            ax_cpv.scatter(
                    timestamps,
                    central_pixel_variations,
                    s=1,
                    c='C1',
                    label = 'Central Subfield Counts'
            )

            ax_cpv.set_ylim(
                    np.nanmean(single_pixel_variations) - 3*np.nanstd(single_pixel_variations),
                    np.nanmean(single_pixel_variations) + 3*np.nanstd(single_pixel_variations)
            )

            ax_cpv.set_xlim(
                    timestamps[0],
                    timestamps[-1]
            )
            ax_cpv.set_ylabel("Central Variations")
            ax_cpv.legend(
                    loc = 'lower left',
                    scatterpoints = 10
            )
            ax_cpv.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            for tick in ax_cpv.get_xticklabels():
                tick.set_rotation(30)
            
            ####

            ax_rms = fig.add_subplot(gs[4,7:])
            ax_rms.scatter(
                    timestamps,
                    root_mean_square,
                    s=1,
                    c='k',
                    label = 'RMS Full'
            )

            ax_rms.scatter(
                    timestamps,
                    central_root_mean_square,
                    s=1,
                    c='C1',
                    label = 'RMS Center'
            )

            ax_rms.set_ylim(
                    np.nanmean(root_mean_square) - 3*np.nanstd(root_mean_square),
                    np.nanmean(root_mean_square) + 3*np.nanstd(root_mean_square)
            )

            ax_rms.set_xlim(timestamps[0],timestamps[-1])
            ax_rms.set_ylabel("RMS")
            ax_rms.set_title("Image Metrics")
            ax_rms.set_xticks([])
            ax_rms.legend(loc = 'lower left',scatterpoints = 10)

            ####

            ax_immn = fig.add_subplot(gs[5,7:])
            ax_immn.scatter(
                    timestamps,
                    image_mean,
                    s=1,
                    c='k',
                    label = 'Image Mean'
            )

            ax_immn.set_ylim(
                    np.nanmean(image_mean) - 3*np.nanstd(image_mean),
                    np.nanmean(image_mean) + 3*np.nanstd(image_mean)
            )
            ax_immn.set_xlim(timestamps[0],timestamps[-1])
            ax_immn.legend(loc = 'lower left',scatterpoints = 10)
            ax_immn.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            for tick in ax_immn.get_xticklabels():
                tick.set_rotation(30)

            ####

            ax_bad = fig.add_subplot(gs[:,:4])
            ax_bad.imshow(bad_frame)
            ax_bad.set_title("Quality 1sigma worse than Mean",pad=25)
            ax_bad.set_ylabel("Pixel Number")

            ax_med = fig.add_subplot(gs[:,4:8])
            ax_med.imshow(okay_frame)
            ax_med.set_title("Quality of Median",pad=25)
            ax_med.set_xlabel("Pixel Number")

            ax_good = fig.add_subplot(gs[:,8:])
            ax_good.imshow(good_frame)
            ax_good.set_title("Quality 1sigma better than Mean",pad=25)
            
            fig.suptitle(filter_name + " Image Quality " + timestamps[0].astype(str),y = 0.7)
            fig.savefig(savestr,bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close('all')
