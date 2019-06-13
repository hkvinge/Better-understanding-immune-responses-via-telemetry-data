import sys

PREFIX = '/home/katrina/a/aminian/Better-understanding-immune-responses-via-telemetry-data/'
tools = 'online_mset/'

sys.path.append(PREFIX)
sys.path.append(PREFIX + tools)

import load_csvs
import tde
import mset

import numpy as np
from matplotlib import pyplot

tempdict,df = load_csvs.load()

delay = 6*60
thresh = 0.05
d=3

mouse='CC004-021'
#mouse='CBA-218'

####

x = tempdict[mouse]

x -= sum(x[:(d-1)*delay])/((d-1)*delay)

# time delayed embedding based on analytical 
# zero-autocorrelation time of pi/2 for a sinusoid.
#    delay = int((np.pi/2)/(t[1] - t[0]))
X = tde.tde(x, delay=delay, nd=d)

# code demands data in X arranged as columns.
# note: code updated to give this as the output of tde.tde().
#    X = X.T

norms = mset.online_mset(X, output_norms=True, thresh=thresh, verbosity=1)

t = np.arange(len(x))

t_d = t[2*delay:]


# get locations of anomalies.
anomalies = (norms>=thresh)
where = np.where(anomalies)[0]
where += 2*delay

anom_windowed = np.convolve(anomalies, np.ones(delay//2)/(delay/2.), mode='same')


