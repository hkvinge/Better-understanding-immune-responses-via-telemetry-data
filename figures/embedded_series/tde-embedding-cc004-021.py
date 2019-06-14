import sys
import pickle
import numpy as np
from matplotlib import pyplot
from scipy import signal # need to smooth dat

sys.path.append('../../online_mset/')
import tde
import mset
f = open('../anomaly_detection_comparison/cc004-021-mset-data.pkl', 'rb')
blob = pickle.load(f)
f.close()
from mpl_toolkits.mplot3d import Axes3D

import cmocean

pyplot.ion()

X = tde.tde( blob['x'], delay=6*60, nd=3 )
t_d = blob['mset_t_d']

tinf = 10000

xs = signal.medfilt(blob['x'], 181)
Xs = tde.tde(xs, delay=6*60, nd=3)


# Colors where center of colormap corresponds to infection
colors_toi = t_d - tinf
colors_toi = colors_toi/(colors_toi.max() - colors_toi.min())
colors_toi += 0.5



#################################

# version 1

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

cax = ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors_toi[::15], s=50, cmap=cmocean.cm.balance, vmin=0, vmax=1)

cbar = fig.colorbar(cax)

# colorbar-relative increments for days.
delcbar = 1440/(t_d.max() -t_d.min())

cbar.set_ticks(0.5 + delcbar*np.arange(-6,6))
cbar.set_ticklabels(np.arange(-6,6))
fig.tight_layout()
ax.set_xlabel(r'$x(t)$', fontsize=14)
ax.set_ylabel(r'$x(t-\tau)$', fontsize=14)
ax.set_zlabel(r'$x(t-2\tau)$', fontsize=14)
fig.tight_layout()

#fig.show()


fig.savefig('embedded_cc004-021-toi.png', dpi=120, bbox_inches='tight')
fig.savefig('embedded_cc004-021-toi.pdf', dpi=120, bbox_inches='tight')

#######################################

# version 2 - work with time of day. 
# use cyclic colormap and corresponding 
# mod-1440 time of day.

# colorbar-relative increments for days.

colors_tod = t_d
colors_tod = np.mod( colors_tod, 1440. )/1440.
#colors_tod = colors_tod/(colors_tod.max() - colors_tod.min())
#colors_tod += 0.5

#colors_tod = np.mod(colors_tod, 1)

fig2 = pyplot.figure(figsize=(8,6))
ax2 = fig2.add_axes([0.05,0.05,0.8,0.9], projection='3d')

cax2 = ax2.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors_tod[::15], s=50, cmap=pyplot.cm.twilight, vmin=0, vmax=1)

#cbar2 = fig2.colorbar(cax2)

##########################################################
#
# Crazy stuff incoming

th = np.linspace(0,2*np.pi, 256)
x0 = np.cos(th)
x1 = np.sin(th)

circle_col = pyplot.cm.twilight( np.linspace(0,1,256) )

ax2o = fig2.add_axes([0.65,0.65,0.25,0.25])
for i in range(len(th)-1):
    ax2o.plot(x0[i:i+2], x1[i:i+2], c=circle_col[i], lw=10)
#
ax2o.axis('square')
ax2o.set_facecolor([1,1,1,0.8])
#ax2o.axis('off')
for s in ax2o.spines.values():
    s.set_visible(False)
#
ax2o.set_xticks([])
ax2o.set_yticks([])
ax2o.set_xlim([-2.,1.5])
ax2o.set_ylim([-1.5,1.5])


# first time index in the embedded plot should 
# correspond to 10am plus two delays, so 10am+6hr+6hr = 10pm.
# colors_tod[0] is 0.5; t_d[0] is 720. 
# this should be associated with the darkest color 
# in the cyclic colormap.
#
# go counterclockwise from theta=0; four labels.
labels = ['10am','4pm','10pm','4am']
displ = 1.2
annotesxy = [[displ,0],[0,displ],[-displ,0],[0,-displ]]
haligns = ['left','center','right','center']
valigns = ['center','bottom','center','top']
rots = [0,0,0,0]

for l,anxy,ha,va,rot in zip(labels,annotesxy,haligns,valigns, rots):
    ax2o.annotate(l,anxy,ha=ha,va=va, fontsize=12, rotation=rot)


##########################################################

#cbar.set_ticks(0.5 + delcbar*np.arange(-6,6))
#cbar.set_ticklabels(np.arange(-5,6))
#fig2.tight_layout()
ax2.set_xlabel(r'$x(t)$', fontsize=14)
ax2.set_ylabel(r'$x(t-\tau)$', fontsize=14)
ax2.set_zlabel(r'$x(t-2\tau)$', fontsize=14)

#fig2.tight_layout()

#fig2.show()


fig2.savefig('embedded_cc004-021-tod.png', dpi=120, bbox_inches='tight')
fig2.savefig('embedded_cc004-021-tod.pdf', dpi=120, bbox_inches='tight')

