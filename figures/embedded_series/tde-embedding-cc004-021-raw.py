import sys
import pickle
import numpy as np
from matplotlib import pyplot
sys.path.append('../../online_mset/')
import tde
import mset
f = open('../anomaly_detection_comparison/cc004-021-mset-data.pkl', 'rb')
blob = pickle.load(f)
f.close()
from mpl_toolkits.mplot3d import Axes3D
blob.keys()
X = tde.tde( blob['x'], delay=6*60, nd=3 )
X.shape
X.min()
X.max()
fig = pyplot.figure()
pyplot.ion()
fig.show()
ax = fig.add_subplots(111, projection='3d')
ax = fig.add_subplot(111, projection='3d')
tinf = 10000
import cmocean
blob.keys()
colors = cmocean.cm.balance( np.array( blob['mset_t_d']>=tinf, dtype=float ) )
ax.scatter(X[0],X[1],X[2], c=colors, s=10, alpha=0.5)
fig.show()
t_d = blob['mset_t_d']
ax.cla()
colors = cmocean.cm.balance( (t_d - t_d.min())/(t_d.max() - t_d.min()) )
ax.scatter(X[0],X[1],X[2], c=colors, s=20, alpha=0.2)
ax.cla()
ax.scatter(X[0][::15],X[1][::15],X[2][::15], c=colors, s=20, alpha=0.2)
ax.scatter(X[0][::15],X[1][::15],X[2][::15], c=colors[::15], s=20, alpha=0.2)
ax.cla()
ax.scatter(X[0][::15],X[1][::15],X[2][::15], c=colors[::15], s=20, alpha=0.2)
ax.cla()
ax.scatter(X[0][::15],X[1][::15],X[2][::15], c=colors[::15], s=40, alpha=0.2)
ax.plot(X[0][:tinf:15],X[1][:tinf:15],X[2][:tinf:15], c=colors[0], lw=2)
ax.plot(X[0][tinf::15],X[1][tinf::15],X[2][tinf::15], c=colors[-1], lw=2)
Out[39].remove()
Out[39][0].remove()
Out[40][0].remove()
fig.tight_layout()
from scipy import signal
xs = signal.medfilt(x, 61)
xs = signal.medfilt(blob['x'], 61)
Xs = tde.tde(xs, delay=6*60, nd=3)
Xs.shape
ax.cla()
ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors[::15], s=80, alpha=0.4)
ax.plot(Xs[0][:tinf:15],Xs[1][:tinf:15],Xs[2][:tinf:15], c=colors[0], alpha=0.4)
ax.plot(Xs[0][tinf::15],Xs[1][tinf::15],Xs[2][tinf::15], c=colors[-1], alpha=0.4)
Out[53][0].remove()
Out[52][0].remove()
ax.plot(Xs[0][:tinf:15],Xs[1][:tinf:15],Xs[2][:tinf:15], c=colors[0], alpha=0.2, lw=4)
xs = signal.medfilt(blob['x'], 181)
Xs = tde.tde(xs, delay=6*60, nd=3)
ax.cla()
ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors[::15], s=80, alpha=0.4)
colors = t_d-t_d.min()
colors += (t_d.max() - t_d.min())/2.
colors += (colors.max() - colors.min())/2.
colors = colors + (colors.max() - colors.min())/2.
colors
colors = colors/(colors.max() - colors.min())
ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=cmocean.cm.balance(colors)[::15], s=80, alpha=0.4)
ax.cla()
ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=cmocean.cm.balance(colors)[::15], s=80, alpha=0.4)
ax.cla()
t_d
t_d[9280]
colors[9280]
colors = t_d-t_d.min()
colors[9280]
colors = t_d - tinf
colors[9280]
colors = colors/(colors.max() - colors.min())
colors[9280]
colors[-1]
colors += 0.5
colors.min()
colors.max()
colors[9280]
ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=cmocean.cm.balance(colors)[::15], s=80, alpha=0.4)
fig.colorbar(Out[85])
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors[::15], s=80, alpha=0.4, cmap=cmocean.cm.balance, vmin=0, vmax=1)
fig.colorbar(cax)
cbar = Out[90]
(t_d - tinf).max()
7772/1440
(t_d - tinf).min()
9280/1440
1440/(t_d.max() -t_d.min())
delcbar = 1440/(t_d.max() -t_d.min())
cbar.set_xticks(0.5 + np.arange(0,1,delcbar))
cbar.set_ticks(0.5 + np.arange(0,1,delcbar))
cbar.set_ticks(0.5 + np.arange(-1,1,delcbar))
cbar.set_ticks(0.5 + delcbar*np.arange(-6,6))
cbar.set_ticklabels(np.arange(-6,6))
cbar.set_ticks(0.5 + delcbar*np.arange(-5,6))
cbar.set_ticks(0.5 + delcbar*np.arange(-6,6))
cbar.set_ticklabels(np.arange(-5,6))
fig.tight_layout()
cbar.grid(None)
cbar.set_ticks?
cbar.set_ticklabels?
cbar.add_lines?
cbar.add_lines(None)
cbar.colorbar
cbar.alpha = 1
cbar.apha
cbar.alpha
cbar.set_alpha(1)
cbar.set_alpha(0.1)
pyplot.ion()
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors[::15], s=50, cmap=cmocean.cm.balance, vmin=0, vmax=1)
ax.cla()
cax = ax.scatter(Xs[0][::15],Xs[1][::15],Xs[2][::15], c=colors[::15], s=50, cmap=cmocean.cm.balance, vmin=0, vmax=1)
colors[9280]
cbar = fig.colorbar(cax)
cbar.set_ticks(0.5 + delcbar*np.arange(-6,6))
cbar.set_ticklabels(np.arange(-5,6))
fig.tight_layout()
ax.set_xlabel(r'$x(t)$', fontsize=14)
ax.set_ylabel(r'$x(t-\tau)$', fontsize=14)
ax.set_zlabel(r'$x(t-2\tau)$', fontsize=14)
fig.tight_layout()
fig.savefig('embedded_cc004-021.png', dpi=120, bbox_inches='tight')
fig.savefig('embedded_cc004-021.pdf', dpi=120, bbox_inches='tight')
%history -f tde-embedding-cc004-021-raw.py
