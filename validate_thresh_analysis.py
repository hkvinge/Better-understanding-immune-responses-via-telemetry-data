import mset_all

import numpy as np
from matplotlib import pyplot

import study_params

thresh = mset_all.thresh
all_norms = mset_all.mset_all()

validate_norms = [all_norms[j] for j in study_params.validate]

##################
#
# Part 2 - the analysis on the training data shows that 
# a threshold of 12 - corresponding to 768 minutes or roughly 
# 12~13 hours - indicates a sufficient gap to mark the end of 
# training while minimizing the potential for false positives
# and not being too stringent that the algorithm can't 
# exit the training phase.
#

thresh_gap = 12
thresh_gaps = [thresh_gap]

# Want to know
#   1. Number of time series thrown out (how many can't exit training?)
#   2. More precisely what the false positive rates are before first true positive?
#   3. Time of first post-infection anomaly?
#   4. Any more detailed observations that can be made?
#   5. What about those "control" mice?
#

results = []

for jj,d in enumerate( validate_norms ):
    single_result = {}
    uhohs = np.where(d>thresh)[0]
    duhohs = np.diff(uhohs)

    single_result['updates'] = 2*360 + 64*uhohs
    single_result['uhohs'] = uhohs
    single_result['duhohs'] = duhohs
    single_result['thresh_gap'] = {}


#   ######
    t = thresh_gap


    thing = {}
    end_train_idx = np.where(duhohs>t)[0]
    if len(end_train_idx)==0:
        thing['train_cutoff'] = np.nan
        thing['error'] = 1
        single_result['thresh_gap'][t] = thing
        continue
    else:
        thing['error'] = 0
        eti = end_train_idx[0] + 1
        thing['train_cutoff'] = eti
    #
    anomalies = uhohs[eti:]
    
    real_time_anomalies = 2*360 + 64*anomalies
    false_positives = sum(real_time_anomalies < 10000.)
    true_positives = sum(real_time_anomalies >= 10000.)
    
    if true_positives>0:
        first_true_positive = anomalies[(real_time_anomalies>=10000)[0]]
    else:
        first_true_positive = np.nan
    #
    thing['false_positives'] = real_time_anomalies[real_time_anomalies < 10000.]
    thing['true_positives'] = real_time_anomalies[real_time_anomalies >= 10000.]

    thing['nfp'] = false_positives
    thing['ntp'] = true_positives
    
    single_result['thresh_gap'][t] = thing

    results.append( single_result )
#

# Now do analysis!
all_fps = []
all_tps = []
all_failures = []

for j,t in enumerate(thresh_gaps):
    fps=[]
    tps=[]
    failures=[]
    for i,r in enumerate(results):
        if r['thresh_gap'][t]['error']>0:
            failures.append(i)
        else:
            fps.append( r['thresh_gap'][t]['nfp'] )
            tps.append( r['thresh_gap'][t]['ntp'] )
        #
    #
    all_fps.append( fps )
    all_tps.append( tps )
    all_failures.append( failures )
#

fp_meds = [np.median(fpi) for fpi in all_fps]
fp_pctiles = np.array( [np.quantile(fpi,[0.1,0.9]) for fpi in all_fps] )


tp_meds = [np.median(tpi) for tpi in all_tps]
tp_pctiles = np.array([np.quantile(tpi,[0.1,0.9]) for tpi in all_tps])

first_tp_times = []
for j,t in enumerate(thresh_gaps):
    fttp = []
    for i,r in enumerate(results):
        if r['thresh_gap'][t]['error']>0:
            continue
        else:
            fttp.append( r['thresh_gap'][t]['true_positives'].min()-10000. )
        #
    #
    first_tp_times.append( fttp )
#

fttp_meds = [np.median(fttp) for fttp in first_tp_times]
fttp_pctiles = np.array([np.quantile(fttp,[0.1,0.9]) for fttp in first_tp_times])

