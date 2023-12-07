"""
Fix for seeing files that had level-0 data moved to cold storage
"""

import numpy as np
import glob

base = "/sunspot/solardata/"
dirs = [
    base+"2019/04/08/", 
    base+"2021/01/18/", 
    base+"2020/09/29/", 
    base+"2021/05/01/", 
    base+"2021/05/03/",
    base+"2021/05/06/",
    base+"2021/10/05/"
]
filters = ['4170', 'gband', 'cak', '3500']
for i in range(len(dirs)):
    print(dirs[i])
    for band in filters:
        seeingFiles = sorted(
                glob.glob(
                    dirs[i]+"*"+band+"_seeing_quality*"
                )
        )
        if len(seeingFiles) == 0:
            continue
        if band == "3500":
            for file in seeingFiles:
                os.remove(file)
        for file in seeingFiles:
            try:
                params = np.load(file)
                continue
            except:
                params = np.load(file, allow_pickle=True).flat[0]
            time = np.array(params['Timestamp'], dtype='datetime64')
            hsm = np.array(params['HSM'])
            hsmc = np.array(params['Central HSM'])
            cpix = np.array(params['Central Pixel Value'])
            csub = np.array(params['Sum of Image Center'])
            immn = np.array(params['Image Mean Value'])
            rms = np.array(params['RMS of Image'])
            crms = np.array(params['RMS of Image Center'])
            arr = np.rec.fromarrays(
                    [
                        time,
                        hsm,
                        hsmc,
                        cpix,
                        csub,
                        immn,
                        rms,
                        crms
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
            np.save(file, arr)

