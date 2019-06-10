
import numpy as np

import calcom

#
prefixes = ['/data4/kvinge/time_series_work/tamu/',
            '/Users/HK/Programming/Calcom/tamu/', 
            '/data3/darpa/tamu/',
            'C:/Users/manuc/Documents/Github/calcom/']

ccd = None
for p in prefixes:
    try:
#        ccd = calcom.io.CCDataSet(p + 'tamu_expts_01-27.h5')
        ccd = calcom.io.CCDataSet(p + 'tamu_expts_01-28.h5')
        break
    except:
        continue
#

if not ccd:
    # failed to find the dataset
    raise Exception('COULD NOT LOAD THE DATASET - ADD THE FOLDER CONTAINING tamu_expts_01-27.h5 TO THE LIST OF PREFIXES IN utils.py')
#

n_mice = len(ccd.data)

# Global variables; updated on calls to process_timeseries()
# To be used for get_labels().
global _mids
global _t_lefts

_mids = []
_t_lefts = []

def load_one(which='t', window=1440, step=np.nan, mouse_id='', reset_globals=True, nan_thresh=120,**kwargs):
    '''
    Loads a timeseries.

    Inputs:
        which : String; 't' for temperature, 'a' for activity (default: 't')
        step : Integer or nan-valued. This indicates the step taken in an individual
            timeseries before a new sample of size window is taken. If np.nan,
            then this defaults to the same value as window (i.e., timeseries are non-overlapping).
            (default: np.nan)
        window : Integer; the size of the windows to be used in terms of
                array size. The underlying units are in minutes. (default: 1440, or one day)
        mouse_id : String; reference to a specific mouse. (default: empty string)
        reset_globals : Boolean; whether to reset the global variables
            _mids and _t_lefts when this is being called. If this is being
            called on its own, you probably want this to be True.
            When this is called by load_all(), it is set False.
            (default: True)
        nan_thresh : integer. Number of NaN values in a chunk of the timeseries that 
            can be tolerated before the chunk is thrown out. (default: 120). 

    Outputs:
        time_chunks : numpy array of dimension n-by-d, where d is the window
            size, and n is the number of timeseries segments salvageable from
            the given mouse. This varies a lot depending on the mouse.

    Notes:
        - If mouse_id is not specified, the global parameter _i is referenced
        and the operations are applied on ccd.data[_i].
        - If the window size is smaller than nan_thresh, then nan_thresh will be 
        internally reset to window-1.
    '''
    import numpy as np
    if kwargs.get('debug',False):
        import pdb
        pdb.set_trace()
    #

    nan_thresh = min(nan_thresh,window-1)

    # get index pointing to appropriate datatype
    datatype = {'t':0, 'a':1, 'b':[0,1]}[which]
    if ( not isinstance(mouse_id,str) ) or ( len(mouse_id)==0 ):
        print('Warning: invalid mouse_id specified. Returning empty array.')
        return np.zeros( (0,window) )
    #

    mid = ccd.find('mouse_id',mouse_id)[0]
    dp = ccd.data[mid]
    tseries = np.array( dp[datatype] )

    # easier to convert the 1-modality case to a 2d array and handle general case.
    if which in ['t','a']:
        tseries.shape = (1,len(tseries))
    #

    len_ts = np.shape(tseries)[1]


    if np.isnan(step):
        step = window

    nchunks = ( len_ts - window )//step +1

    time_chunks = [ 
                    process_timeseries( tseries[:, i*step : i*step + window ], nan_thresh=nan_thresh) 
                    for i in range(nchunks) 
                ]

    t_lefts_local = np.array( [ i*step for i in range(nchunks) ] )
    mouse_pointers_local = np.array( [ mid for _ in range(nchunks) ] )

    # update the globals based on what time_chunks looks like.
    shapes = np.array( [ np.prod( np.shape(tc) ) for tc in time_chunks ] )
    valid_tc = np.where( shapes!=0 )[0]

    t_lefts_local = t_lefts_local[ valid_tc ]
    mouse_pointers_local = mouse_pointers_local[ valid_tc ]

    global _mids
    global _t_lefts

    if reset_globals:
        _mids = mouse_pointers_local
        _t_lefts = t_lefts_local
    else:
        _mids += list( mouse_pointers_local )
        _t_lefts += list( t_lefts_local )
    #


    if not isinstance(datatype,list):
        datatype = [datatype]
    #

    # import pdb
    # pdb.set_trace()

    output = np.zeros( (len(datatype), len(valid_tc), window), dtype=float)

    for j,idx in enumerate(valid_tc):
        for k,dtype in enumerate(datatype):
            #output[k,j,:] = time_chunks[idx][dtype]
            if which=='b':  #I can't do the elegant solution
                output[k,j,:] = time_chunks[idx][k]
            else:
                output[k,j,:] = time_chunks[idx]
        #
    #

    # don't want the more general data structure if 
    # only one type of timeseries was requested; flatten back down.
    if np.shape(output)[0]==1:
        output.shape = (output.shape[1], output.shape[2])

    return output
#

def load_all(which='t', window=1440, step=np.nan, nan_thresh=120,**kwargs):
    '''
    Loads all timeseries by repeatedly calling load_one iteratively.

    Inputs:
        Same optional inputs as load_one, except mouse_id, with same default parameters.
    Outputs:
        time_chunks : numpy array of dimension N-by-d, where the arrays
            from each result are concatenated.
    '''
    import numpy as np

    # Reset the global variables
    global _mids
    global _t_lefts
    _mids = []
    _t_lefts = []

#    combo = [[],[]] # only used for which=='b'
    time_chunks_all = []
    for i in range(n_mice):
        # print(i,ccd.data[i].mouse_id.value)
        time_chunks = load_one(
                            which = which,
                            window = window,
                            step = step,
                            mouse_id = ccd.data[i].mouse_id.value,
                            nan_thresh = nan_thresh,
                            reset_globals = False,
                            **kwargs                        
                        )
        time_chunks_all.append( time_chunks )

    #

    if which=='b':
        time_chunks = np.concatenate( time_chunks_all, axis=1 )
    else:
        time_chunks = np.concatenate( time_chunks_all, axis=0 )
    #

    return time_chunks
#

def process_timeseries(tseries_raw, nan_thresh=120,**kwargs):
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
    import numpy as np

    # import pdb
    # pdb.set_trace()

    verbosity = kwargs.get('verbosity',0)

    dims = np.shape(tseries_raw)
    if len(dims)>1:
        # Are there multiple timeseries?
        m,d = dims[0],dims[1]
    else:
        m = 1
        d = dims[0]
    #

    # how many time points are polluted by nans?
    # This will throw out the datapoint if 
    nan_locs = np.where(tseries_raw!=tseries_raw)
    if m>1:
        nnans = np.unique(nan_locs[1])
    else:
        nnans = np.unique(nan_locs)
    #

    if len(nnans) >= nan_thresh:
        if verbosity!=0:
            print('The timeseries has greater than %i NaNs; throwing it out.'%nan_thresh)
        if m>1:
            return np.vstack( [np.zeros((0,d)) for _ in range(m)] )
        else:
            return np.zeros( (0,d) )
    #

    # Else, make a copy of the timeseries and clean it up in-place.
    tseries = np.array(tseries_raw)

    # It's easier to wrap a single timeseries into a list
    # and handle everything like the general case.
#    if m==1:
#        tseries = [tseries]
    #

    # Check beginning and end of timeseries for nans; replace
    # with the first non-nan value in the appropriate direction.


    for j in range(m):
        active_ts = tseries[j]
        if np.isnan( active_ts[0] ):
            for i in range(nan_thresh):
                if not np.isnan(active_ts[i]):
                    break
            #
            active_ts[:i] = active_ts[i]
        #
        if np.isnan( active_ts[-1] ):
            for i in range(d-1, d-nan_thresh-1, -1):
                if not np.isnan(active_ts[i]):
                    break
            #
            active_ts[i:] = active_ts[i]
        #

        nan_loc = np.where(np.isnan(active_ts))[0]
        if len(nan_loc)==0:
            # Nothing more to be done.
            # return tseries
            continue
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
            fl = active_ts[left]

            right = min(chunk[-1] + 1, d-1)
            fr = active_ts[right]

            m = (fr-fl)/(len(chunk) + 1)
            active_ts[chunk] = fl + m*np.arange(1 , len(chunk)+1)

            count += 1
        #
    #

    if m==1:
        # undo what we did wrapping the one timeseries in a list.
        return tseries[0]
    else:
        return tseries
    #
#

def get_labels(attr):
    '''
    Returns an array of labels for the given attribute,
    AFTER a call to load_all() has been made. Running this
    prior to that will return an error. The reason is that
    the way the data is generated dynamically in
    load_one() and load_all() depends on the window and step
    parameters; global variables in utils are updated to
    keep track of information necessary specific to that
    partitioning of the data.

    Inputs:
        attr: string. Name of a built-in attribute in the dataset
            (see utils.ccd.attrnames) or one of a small set of
            additional attributes derived from these which are
            of interested to us (e.g. t_translated, t_since_infection).
    Outputs:
        labels: array of appropriate datatype for the requested information.
            The size will be commensurate to the data generated with
            the most recent call to utils.load_all(); e.g. if
            you have a data array of shape (12345,1440) then
            the labels will be shape (12345,).

    The list of additional attributes (as of February 7, 2019) are:
        t_translated : integer; (time since start of experiment) - (time of infection).
            The time since start of experiment is measured
            on the left endpoint of the chunk.
        t_post_infection : integer; max(0, t_translated). This presumes
            all pre-infection data is of the same quality.
        infected : 0/1 integer. This asks if the chunk is at a point 
            pre- or post-infection. Essentially (t_translated > 0) casted to integer.
    '''
    import numpy as np

    global _mids
    global _t_lefts

    if attr in ccd.attrnames:
        labels = ccd.get_attr_values(attr, idx=_mids)

    elif attr in ['t_translated', 't_post_infection', 'infected']:

        itimes = ccd.get_attr_values('infection_time', idx=_mids)
        t_translated = np.array(_t_lefts) - itimes

        if attr=='t_translated':
            labels = t_translated
        elif attr=='t_post_infection':
            labels = np.maximum(0., t_translated)
        elif attr=='infected':
            labels = np.array( t_translated > 0., dtype=int)
        #
    else:
        raise ValueError('Attribute %s not recognized. See the docstring for a list of allowable inputs.'%str(attr))
    #

    return labels
#

def generate_mask(**kwargs):
    '''
    Purpose: generate a mask to be used to remove datapoints we expect to perform poorly 
    for reasons outside our control. 
    MUST BE RUN AFTER LOAD_ALL().

    Includes:
        1. experiment_11 : these are mock infections.
        2. If the time of infection lies inside the time chunk.

    Inputs:
        None
    Outputs:
        mask : array with True/False entries, which masks out 
            entries deemed undesirable.
    Optional inputs:
        update_globals : Boolean. Whether or not to automatically mask out 
            the global variables in utils; these are utils._mids and utils._t_lefts.
            Note that these are used to generate labels and so on; if not 
            automatically masked you need to do it yourself.
            (default: True)
    '''
    update_globals = kwargs.get('update_globals', True)

    global _mids
    global _t_lefts

    # First - remote experiment_11.
    mask = np.array( np.ones(len(_mids)), dtype=bool )
    expt_vals = get_labels('experiment')

    locs = np.where(expt_vals=='experiment_11')[0]

    mask[locs] = False
    
    # Next - identify "boundary" timeseries based on _t_lefts
    # and time of infections pulled from data.
    itimes = get_labels('infection_time')
    mo = np.array(_t_lefts)

    # infer the window size then use that to judge if there's an overlap.
    window = mo[1] - mo[0]
    for k in range(len(mask)):
        if ( mo[k] - itimes[k] )*( mo[k] - itimes[k] + window )<0:
            mask[k] = False
        #
    #
    
    if kwargs.get('update_globals',True):
        _mids = np.array(_mids)[mask]
        _t_lefts = np.array(_t_lefts)[mask]
    #

    return mask
#

def generate_partitions(attrvals):
    '''
    Purpose: generate partitions for a **one-attribute-out** cross-validation.

    Inputs: 
        labels - a list generated by get_labels()
            upon which to do a one-attribute-out cross-validation.
            NOT the classification.

    Outputs: 
        partitions - a list of pairs of lists. The first entry is the 
            indices of the training set; the second entry is the indices 
            of the test set.    
    '''
    all_idx = np.arange(len(attrvals))

    eq = {k:np.where(np.array(attrvals)==k)[0] for k in np.unique(attrvals)}
    partitions = [ [np.setdiff1d(all_idx, eq[k]), eq[k]] for k in eq.keys() ]
    
    return partitions
#

def line2pheno(line):
    '''
    Input: string, the line of the mouse
    Output: expected phenotype; latest classifications by collaborators
    '''
    if line in ['C57B6','CC025','CC042','CC013']:
        return 'sensitive'
    elif line in ['CC001','CC006','CC011','CC012','CC019','CC043','CC057','CC043','CC015']:
        return 'tolerant'
    elif line in ['CC002','CC004','CC041','CC051']:
        return 'resistant'
    else:
        return 'unknown'
#

if __name__=="__main__":
    # Temporary testing
    from matplotlib import pyplot

    npoints = 50

    tseries_bad = np.random.choice(np.arange(-5.,6.), npoints)
    tseries_bad[4] = np.nan         # single point in the middle
    tseries_bad[-5:] = np.nan       # range at the tail of the timeseries
    tseries_bad[:2] = np.nan        # range at the head of the timeseries
    tseries_bad[7:10] = np.nan      # small range in the middle
    tseries_bad[11:21] = np.nan     # wider range in the middle

    tseries = process_timeseries(tseries_bad)

    fig,ax = pyplot.subplots(1,1)

    ax.plot( np.arange(npoints), tseries, c='r', marker='.', markersize=10, label='interpolated')
    ax.scatter( np.arange(npoints), tseries_bad, marker='o', edgecolor='k', s=100, label='original data')

    ax.legend(loc='upper right')
    fig.show()
#
