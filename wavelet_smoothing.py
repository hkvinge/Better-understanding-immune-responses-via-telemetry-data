from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import packages and functions
import sys
import math
sys.path.append('/data3/darpa/calcom/')
import calcom
import random
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy
import pywt

# Compression gives the degree of detail coefficients
# we save when applying wavelet smoothing. The larger
# this integer, the more smoothing is done
compression = 4

# Use Calcom to load time series

# path = '/data4/kvinge/time_series_work/tamu/'
# ccd = calcom.io.CCDataSet(path + 'tamu_expts_01-27.h5')

# Note - utils also loads the dataset, and has functionality to 
# search in different directories for the file.
ccd = utils.ccd

# Initialize in which to store the time series
ts = []

# Convert time series into a list of numpy arrays
kept_idx = []
for count,i in enumerate(ccd.data):
    a = utils.process_timeseries(i, nan_thresh=60*4)
    if len(a) > 0:
        a = np.array(a)
        # Use only the temperature values
        ts.append(a[0,:])
        kept_idx.append( count )
#
        
# We will calculate two types of smoothings. One type stored in
# ts_smooth will be on the same scale as the original time series. 
# That is, it will have the same number of time steps. The other, 
# which includes the work 'shrunk' in variable names, actually 
# shrinks the scale. Which is better to use depends on the application.

# List of smoothed time series with the same time scale
ts_smooth = []
# List of smoothed time series with scale shrunk
ts_smooth_shrunk = []

# Iterate through all time series and smooth them
for count, a in enumerate(ts):
    a_approx_shrunk = a
    # Iteratively calculate detail coefficients
    for j in range(compression):
        a_approx_shrunk, a_detail = pywt.dwt(a_approx_shrunk,'db1')
    # Save the resulting `smoothed' time series on a reduced scale
    ts_smooth_shrunk.append(a_approx_shrunk)
    # Initialize a new variable and use the inverse transform to 
    # iteratively restore the scale but with detail coefficients set
    # to zero.
    c = a_approx_shrunk
    for j in range(compression):
        c = pywt.idwt(c,np.zeros(np.shape(c)),'db1')
    ts_smooth.append(c)
#

if __name__=="__main__":
    # Plot some samples to see what they look like
    sample_temperatures = ts[0]
    plt.plot(np.transpose(sample_temperatures))
    sample_temp_smooth = ts_smooth[0]
    plt.plot(np.transpose(sample_temp_smooth))
    plt.show()
    sample_temp_smooth_shrunk = ts_smooth_shrunk[0]
    plt.plot(np.transpose(sample_temp_smooth_shrunk))
    plt.show()
#