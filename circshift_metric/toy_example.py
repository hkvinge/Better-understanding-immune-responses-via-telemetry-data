import numpy as np
from matplotlib import pyplot


def d_c(x,y):
    # vanilla cosine (dis)similarity.
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    return 1. - np.dot( x-xbar, y-ybar )/(np.linalg.norm(x-xbar)*np.linalg.norm(y-ybar))
#

def d_qc(x,y, return_shift=False):
    # minimum over cosine (dis)similarities when considering circular shifts 
    # of the vectors. Basically, we'd like to "mod out" phase shifts.
    all_dist = np.array([ d_c( np.roll(x,j), y ) for j in range(len(x)-1) ])
    if return_shift:
        return min(all_dist), np.argmin(all_dist)
    else:
        return min(all_dist)
#


eps = 0.1           # noise
phase = np.pi/2     # inserted phase difference

t = np.linspace(0,7*np.pi, 1001)

x = np.sin(t) + eps*np.random.randn(len(t))
y = np.sin(t-phase) + eps*np.random.randn(len(t))

#

fig,ax = pyplot.subplots(2,1, sharex=True, figsize=(10,6))

#
# Visualize original time series
#
sim_c = d_c(x,y)

ax[0].plot(t,x, label=r'$x(t)$')
ax[0].plot(t,y, label=r'$y(t)$')

ax[0].legend(loc='upper right')

ax[0].text(0.05,0.9,r'$d_c(x,y) = %.2f$'%sim_c, ha='left', va='top', fontsize=16, transform=ax[0].transAxes, bbox=dict(facecolor='w', alpha=0.9))

#
# Visualize time series aligned by circular shifting.
#
sim_qc,shift = d_qc(x,y, return_shift=True)

ax[1].plot(t, np.roll(x,shift), label=r'Circular shift of $x(t)$')
ax[1].plot(t, y, label=r'$y(t)$')

ax[1].legend(loc='upper right')

ax[1].text(0.05,0.9,r'$d_{qc}(x,y) = %.2f$'%sim_qc, ha='left', va='top', fontsize=16, transform=ax[1].transAxes, bbox=dict(facecolor='w', alpha=0.9))

fig.tight_layout()
fig.show()

#fig.savefig('circshift_cosine_similarity_ex.png', dpi=120, bbox_inches='tight')

pyplot.ion()

