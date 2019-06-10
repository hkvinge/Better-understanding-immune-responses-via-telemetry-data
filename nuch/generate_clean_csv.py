# script to generate clean dataset, throwing
# out some timeseries with large blocks of missing data.

import numpy as np
import utils
from scipy import signal as spsi
import multiprocessing
import csv

smoothing_window = 31  # needs to be odd for scipy.signal.medfilt()

ccd = utils.ccd

bads = []
goods = []
imputed_temps = []
for j,dp in enumerate(ccd.data):
    itime = dp.infection_time.value
    pis = dp.post_inoc_sac.value
    stop = itime+pis
    if stop<9000:
        bads.append(j)
        continue
    #
    result = utils.process_timeseries(dp[:,:itime+pis], verbosity=1)
    
    if np.shape(result)[0]==0:
        bads.append(j)
    else:
        goods.append(j)
        imputed_temps.append(result[0])
#

def medfiltme(thingy):
    output = spsi.medfilt( thingy, kernel_size=smoothing_window )
    return output
#

print('beginning median filter smoothing')
final_temps = []
# for some reason, median filter is really, really slow.
#
for j,t in enumerate(imputed_temps):
    final_temps.append( medfiltme(t) )
    print('%i of %i'%(j+1,len(imputed_temps)), end='\r')
#

#p = multiprocessing.Pool(48)
#final_temps = p.map(medfiltme, imputed_temps)



# 
df = ccd.generate_metadata()    # note - requires pandas.

expts = ccd.get_attrs('experiment', idx=goods)
mids = ccd.get_attrs('mouse_id', idx=goods)

# create binary attribute for mice that actually weren't infected
mock = (expts=='experiment_11')
phenos = [utils.line2pheno(m.split('-')[0]) for m in mids]

df = df.iloc[goods]
df['mock'] = mock
df['phenotype'] = phenos


df.to_csv('tamu_01-28_metadata_clean.csv', index=None)

with open('tamu_01-28_temps_smooth.csv','w') as f:
    print('writing imputed smoothed temperature timeseries...')
    csvw = csv.writer(f)
    for j,(mid,temp) in enumerate( zip(mids, final_temps) ):
        if hasattr(temp,'__iter__'):    # some weird bug with one of the timeseries
            row = [mid] + list(temp)
            csvw.writerow(row)
            print('%i of %i'%(j+1,len(final_temps)), end='\r')
#

print('\n\n\t~~~~~~~~~~~ DONE ~~~~~~~~~~~\n')

