'''
A utility module for preparing and executing laplacian eigenmaps
on mice timeseries
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import signal

def process_timeseries(tseries_raw, nan_thresh=99999,**kwargs):
    '''
    Applies preprocessing to a timeseries based on the rules:

    1. If the timeseries has missing data above a threshold,
        the datapoint is thrown out in the sense that
        an empty array of commensurate second dimension
        is returned.
    2. Else, NaNs in the data are filled by linear interpolation
        of neighboring data.

    Inputs:
        tseries_raw: a numpy array shape (d,), possibly containing NaNs.
    Optional inputs:
        nan_thresh: integer. If there are at least this many nans, nothing
            is done; instead an empty array of shape (0,d) is returned. (default: 120)
        verbosity: integer; 0 indicates no output. (default: 0)
    Outputs:
        tseries: a numpy array shape either (d,) or (0,d), depending on
            whether the timeseries was thrown out for having too much
            missing data.
    '''

    verbosity = kwargs.get('verbosity',0)

    d = len(tseries_raw)
    nnans = np.where(tseries_raw!=tseries_raw)[0]

    if len(nnans) >= nan_thresh:
        if verbosity!=0:
            print('The timeseries has greater than %i NaNs; throwing it out.'%nan_thresh)
        return np.zeros( (0,d) )
    #

    # Else, make a copy of the timeseries and clean it up in-place.
    tseries = np.array(tseries_raw)

    # Check beginning and end of timeseries for nans; replace
    # with the first non-nan value in the appropriate direction.
    if np.isnan( tseries[0] ):
        for i in range(nan_thresh):
            if not np.isnan(tseries[i]):
                break
        #
        tseries[:i] = tseries[i]
    #
    if np.isnan( tseries[-1] ):
        for i in range(d-1, d-nan_thresh-1, -1):
            if not np.isnan(tseries[i]):
                break
        #
        tseries[i:] = tseries[i]
    #

    nan_loc = np.where(np.isnan(tseries))[0]
    if len(nan_loc)==0:
        # Nothing to be done.
        return tseries
    #

    # Now work in the interior. Identify locations of
    # nans, identify all contiguous chunks, and fill them
    # in one by one with the nearest non-nan values.
    contig = []
    count = 0
    while count < len(nan_loc):
        active = nan_loc[count]
        # scan the following entries looking for a continguous
        # chunk of nan.
        chunk = [nan_loc[count]]
        for j in range(count+1,len(nan_loc)):
            if (nan_loc[j] == nan_loc[j-1] + 1):
                chunk.append( nan_loc[j] )
                count += 1
            else:
                break
        #

        # Identify the tether points for the linear interpolation.
        left = max(chunk[0] - 1, 0)
        fl = tseries[left]

        right = min(chunk[-1] + 1, len(tseries)-1)
        fr = tseries[right]

        m = (fr-fl)/(len(chunk) + 1)
        tseries[chunk] = fl + m*np.arange(1 , len(chunk)+1)

        count += 1
    #

    return tseries


def most_consec_nans(vec, step = 1):
    """
    Returns length of string with most consecutive nan values
    Input: 1D array
    Output: Integer
    """
    nan_loc = np.where(vec != vec)[0].tolist()
    bunch = []
    result = [bunch]
    expect = None
    for v in nan_loc:
        if (v == expect) or (expect is None):
            bunch.append(v)
        else:
            bunch = [v]
            result.append(bunch)
        expect = v + step
    max_len = 0
    for string in result:
        if len(string) > max_len:
            max_len = len(string)
    return max_len


class MyMice():
    """
    A class to play with the mice data
    -originally used for laplacian eigenmaps
    """

    def __init__(self, ccd):
        self.ccd = ccd
        self.mice_count = len(self.ccd.data)
        self.inf_times = self.ccd.get_attrs('infection_time')
        self.lines = self.ccd.get_attrs('line')
        self.id_list = self.ccd.get_attrs('_id')
        self.mouse_id_list = self.ccd.get_attrs('mouse_id')
        
        # tweak - replace "C57B6" with "C57BL/6J"
        new_mouse_id_list = []
        import re
        
        pattern = 'C57B6([a-zA-Z0-9\-]{1,})'
        prefix = "C57BL/6J"
        for j,mouse_name in enumerate(self.mouse_id_list):
            hit = re.match(pattern, mouse_name)
            if hit: # did we hit?
                suffix = hit.groups(1)[0]
                # replace
                print('OLD: %s => NEW: %s'%(mouse_name, prefix+suffix))
                new_mouse_id_list.append( prefix+suffix )
            else:
                new_mouse_id_list.append( mouse_name )
        #
        self.mouse_id_list = np.array(new_mouse_id_list)
        
        #
        self.temp_data = self.ccd.data[0:self.mice_count][0]


    def normalize_data(self, type = ""):
        """
        min_max = normalize via x_i = (x_i - min(x_i))/(max(x_i) - min(x_i))
        mean = normalize via x_i = (x_i - mean(x_i)/stdv(x_i))
        """
        if type == "min_max":
            for i in range(self.data.shape[1]):
                self.data[:,i] = (self.data[:,i] - np.min(self.data[:,i]))/(np.max(self.data[:,i]) - np.min(self.data[:,i]))
        elif type == "mean":
            for i in range(self.data.shape[1]):
                if np.std(self.data[:,i]) == 0:
                    self.data[:,i] = self.data[:,i] - np.mean(self.data[:,i])
                else:
                    self.data[:,i] = (self.data[:,i] - np.mean(self.data[:,i])) / np.std(self.data[:,i])
        return self.data

    def get_windowed_data(self, window = 2880, process_data = True, **kwargs):
        """
        -slices data into pre_inf/post_inf based on inf_times and window size
        -processes data by deleting mice that have too many consecutive nans

        optional kwargs: consec_nan_thresh, total_nan_thresh, wiggle_room (for pulling sliding windows)
        """
        consec_nan_thresh = kwargs.get('consec_nan_thresh', 60)
        total_nan_thresh = kwargs.get('total_nan_thresh', 99999)
        wiggle_room = kwargs.get('wiggle_room', 1440)

        self.data = np.empty((2*window, self.mice_count))

        good_mice_idx = []
        for i in range(self.mice_count):
            self.data[:,i] = self.ccd.data[i][0][self.inf_times[i] - window : self.inf_times[i] + window]
            if most_consec_nans(self.data[:,i]) > consec_nan_thresh:
                continue
            else:
                good_mice_idx.append(i)

        if process_data:
            self.data = self.data[:,good_mice_idx]

            # update attributes of data set based on good_mice_idx
            self.mice_count = self.data.shape[1]
            self.inf_times = self.inf_times[good_mice_idx]
            self.lines = self.lines[good_mice_idx]
            self.id_list = self.id_list[good_mice_idx]
            self.mouse_id_list = self.mouse_id_list[good_mice_idx]

            for i in range(len(good_mice_idx)):
                if np.any(self.data[:,i] != self.data[:,i]):
                    self.data[:,i] = process_timeseries(self.data[:,i], total_nan_thresh)

        self.pre_inf_data = self.data[:window,:]
        self.post_inf_data = self.data[window:,:]

        return self.data, self.pre_inf_data, self.post_inf_data

    def get_window(self, win_len = 1440, start = -4320):
        """
        gets one window of specified length. Must be used after self.get_windowed_data

        TODO:  check length of self.data and work off those parameters


        start = start time of window relative to infection time
            time = 0 is Time of Infection
            time = -4320 is 3 days prior
        """
        self.window = np.empty((win_len, self.mice_count))
        for i in range(self.mice_count):
            self.window[:,i] = self.data[start + 4320: start + 4320 + win_len, i]

        return self.window

    def get_sliding_windows(self, win_len = 120, n = 10, normalize = True, days = 6, **kwargs):
        """
        creates 3D array with each 2D array being a window of data.
        n = number of windows between 3 days prior and 3 days after
        win_len = length of each window in minutes (default = 1 day (1440 min))
        days = number of days to slide the window through (default = 6 (3 days prior, 3 days after inf))
        """
        minutes = 1440 * days
        self.sliding_window = np.zeros((n - 1, win_len, self.mice_count))
        for i in range(n - 1):
            self.sliding_window[i,:,:] = self.get_window(win_len = win_len, start = -int(minutes/2) + int(minutes/n)*i)
            if normalize:
                if type == 'mean':
                    for j in range(self.sliding_window[i,:,:].shape[1]):
                        if np.std(self.sliding_window[i,:,j]) == 0:
                            self.sliding_window[i,:,j] = (self.sliding_window[i,:,j] - np.mean(self.sliding_window[i,:,j]))
                        else:
                            self.sliding_window[i,:,j] = (self.sliding_window[i,:,j] - np.mean(self.sliding_window[i,:,j])) / np.std(self.sliding_window[i,:,j])
                elif type == 'minmax':
                    for j in range(self.sliding_window[i,:,:].shape[1]):
                        self.sliding_window[i,:,j]  = (self.sliding_window[i,:,j] - np.min(self.sliding_window[i,:,j])) / (np.max(self.sliding_window[i,:,j]) - np.min(self.sliding_window[i,:,j]))
        return self.sliding_window
