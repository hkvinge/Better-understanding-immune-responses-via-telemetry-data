
import numpy as np
import scipy

def interval_exit_detector(xv,ib,**kwargs):
    '''
    Anomaly detector based on constructing a range of
    acceptable values from the signal xv[:ib]; data
    xb[ib:] is flagged as anomalous if
    xb[ib:]>max(xv[:ib]) OR xb[ib:]<min(xv[:ib]).

    Inputs:
        xv - numpy array dimension n; any time series
        ib - integer; indicating end of "baseline" data.
    Outputs:
        yv - numpy array dimension n; boolean time series
            indicating anomalous points. By construction, 
            there will be no anomalous points in yv[:ib].
            
    Optional inputs:
        ignore_nan - boolean; how np.nan's are handled in
            xv. If True, the corresponding data in yv is
            marked False (not anomalous). Else, the corresponding
            data in yv is marked as np.nan. (default: True)
    '''
    ignore_nan = kwargs.get('ignore_nan', True)

    xvc = np.array(xv)

    baseline = xvc[:ib]
    bmin,bmax = np.nanmin(baseline),np.nanmax(baseline)

    nanloc = np.where(xvc!=xvc)[0]
    if ignore_nan:
        # replace with a baseline value.
        xvc[nanloc] = bmin
    #

    goodloc = (xvc==xvc)

    # Because of how the "where" kwarg works, need to do
    # it this way to map np.nan values to False rather than True.
    # pdb.set_trace()
    ge_min = np.greater_equal(xvc,bmin, where=goodloc)
    le_max = np.less_equal(xvc,bmax, where=goodloc)

    yv = np.logical_not( np.logical_and( ge_min, le_max ) )

    return yv
#

def off_pattern(anomalies):
    '''
    Returns the index of the first anomalous point
    of an "anomaly" array. 
    
    Returns the index of the last entry in the time series
    if the array "anomalies" has no True values.

    Inputs:
        anomalies - numpy boolean array, e.g. from interval_exit_detector()
    Outputs:
        idx - integer pointing to the first True value in anomalies.
    '''
    locs = np.where(anomalies)[0]
    if len(locs)>0:
        loc = locs[0]
    else:
        loc = len(anomalies)-1
    return loc
#

def full_blown_disease(anomalies,**kwargs):
    '''
    Returns the first index where a centered average of the 
    anomalies array exceeds a threshold. By default, this 
    criterion is 50% anomalies in a one-day window, but these 
    values can be changed.

    Inputs:
        anomalies - numpy boolean array.
    Outputs:
        idx - integer pointing to the location of "full blown disease".
            This is judged based on a percentage of anomalous values
            over a given window length.
    Optional inputs:
        thresh - float in (0,1], indicating percentage of True values in
            anomalies within the specified window required to set the alarm. (default: 0.5)
        window - integer, indicating the size of the window to use. (default: 24*60; i.e., one day)
    '''
    thresh = kwargs.get('thresh', 0.5)
    window = kwargs.get('window', 1*24*60)

    pct_ts = np.convolve(anomalies, np.ones(window)/window, mode='valid')

    where = np.where(pct_ts > thresh)[0]
    if len(where)>0:
        loc = where[0]+window
    else:
        loc = len(anomalies)-1
    return loc
#

def mset_feature_vector(anoms, gap_thresh, infection_time=10000, skip_len=64, delay=2*360):
    '''
    Generates a feature vector from a collection of anomalies.
    It's assumed the anomalies come from the embedded, then subsampled, 
    time series, so parameters here are needed to compare back to infection time.
    
    Defunct - moved on to other considerations rather than attempting clustering 
    based on features generated from MSET. Not enough shape is revealed here.
    '''
    if isinstance(gap_thresh, list) or isinstance(gap_thresh, np.ndarray):
        # apply for each value, then concatenate.
        thing = []
        for g in gap_thresh:
            thing.append( mset_feature_vector(anoms, g, infection_time=infection_time, skip_len=skip_len, delay=delay) )
        #
        return np.concatenate(thing)
    else:
    
        uhohs = np.where(anoms)[0]
        duhohs = np.diff(uhohs)
        
        # get the "significant" anomalies - ones appearing after a gap between updates.
        flagged = uhohs[ np.where(duhohs>gap_thresh)[0] + 1 ]
        
        # translate to time relative to inoculation.
        flagged_transformed = delay + skip_len*flagged - infection_time

        flagged_before = np.where(flagged_transformed < 0.)[0]
        flagged_after = np.where(flagged_transformed >= 0.)[0]
        
        if len(flagged_before)==0: print('no anomalies flagged before!')
        if len(flagged_after)==0: print('no anomalies flagged after!')
        
        return np.array([ len(flagged_before), len(flagged_after), min(flagged_transformed), max(flagged_transformed) ])
    #
#
