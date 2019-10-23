import mset_all

import numpy as np
from matplotlib import pyplot

import study_params

thresh = mset_all.thresh
all_norms = mset_all.mset_all()

train_norms = [all_norms[j] for j in study_params.train]

##################
#
# Part 1 - loop over a collection of thresholds which would indicate 
#   the end of the "training" period for the algorithm.
#
#   Note these are in 64 minute increments - so a threshold of 25 is 
#   corresponding to 1600 minutes, or a little more than a day.
#

thresh_gaps = np.arange(1,26)
results = []

for jj,d in enumerate( train_norms ):
    single_result = {}
    uhohs = np.where(d>thresh)[0]
    duhohs = np.diff(uhohs)

    single_result['updates'] = 2*360 + 64*uhohs
    single_result['uhohs'] = uhohs
    single_result['duhohs'] = duhohs
    single_result['thresh_gap'] = {}
    for t in thresh_gaps:

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
    #
    print(jj)
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

#########

fig2,ax2 = pyplot.subplots(3,1, sharex=True, figsize=(12,6))

ax2[0].plot(thresh_gaps, tp_meds, marker='s', label='True positive count')
ax2[0].fill_between(thresh_gaps, tp_pctiles[:,0], tp_pctiles[:,1], alpha=0.1)
ax2[0].plot(thresh_gaps, fp_meds, marker='s', label='False positive count')
ax2[0].fill_between(thresh_gaps, fp_pctiles[:,0], fp_pctiles[:,1], alpha=0.1)

ax2[0].set_ylabel('Count', fontsize=18)
ax2[0].legend(loc='upper right', fontsize=14)
#

ax2[1].plot(thresh_gaps, [len(fa) for fa in all_failures], c='k', marker='.', ms=10, label='Time series failing to exit training')
ax2[1].set_ylabel('Count', fontsize=16)
ax2[1].legend(loc='upper left', fontsize=14)
#


ax2[2].plot(thresh_gaps, fttp_meds, marker='s', c='r', label='Time to first true positive')
ax2[2].fill_between(thresh_gaps, fttp_pctiles[:,0], fttp_pctiles[:,1], alpha=0.1, color='r')
ax2[2].set_ylabel('Days', fontsize=16)
ax2[2].legend(loc='upper left', fontsize=14)
#
for axi in ax2: axi.set_xticks(list(range(0,26,5))+[12])
for axi in ax2: axi.xaxis.grid()

ax2[2].set_yticks(np.arange(0,7*1440+1,1440))
ax2[2].set_yticklabels(np.arange(0,8))

ax2[2].yaxis.grid()

ax2[2].set_xlabel('Threshold to exit training', fontsize=16)

fig2.tight_layout()
fig2.show()

fig2.savefig('mset_train_thresh_performance.png', dpi=120, bbox_inches='tight')
