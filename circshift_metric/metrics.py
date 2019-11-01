def d_c(x,y):
    # vanilla cosine (dis)similarity.
    import numpy as np
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    return 1. - np.dot( x-xbar, y-ybar )/(np.linalg.norm(x-xbar)*np.linalg.norm(y-ybar))
#

def d_qc(x,y, return_shift=False):
    # minimum over cosine (dis)similarities when considering circular shifts 
    # of the vectors. Basically, we'd like to "mod out" phase shifts.
    import numpy as np
    all_dist = np.array([ d_c( np.roll(x,j), y ) for j in range(len(x)-1) ])
    if return_shift:
        return min(all_dist), np.argmin(all_dist)
    else:
        return min(all_dist)
#

def euclidean(x,y):
    import numpy as np
    return np.linalg.norm(x-y,2)
#
