import tde
import mset
import numpy as np
from matplotlib import pyplot

n = 4000
thresh = 0.1

t = np.linspace(0, 10*np.pi, n)
x = np.sin(t) + 0.1*np.random.randn(n)

# introduce a new type of signal halfway through.
heavy = (1 + np.tanh( 0.5*(t-5*np.pi) ))/2.
x += heavy*np.sin(t/2) + heavy*(-np.sin(t))

# time delayed embedding based on analytical 
# zero-autocorrelation time of pi/2 for a sinusoid.
delay = int((np.pi/2)/(t[1] - t[0]))
X = tde.tde(x, delay=delay)

# code demands data in X arranged as columns.
X = X.T

norms = mset.online_mset(X, output_norms=True, thresh=thresh, verbosity=1)

# visualize
fig,ax = pyplot.subplots(3,1, 
                    figsize=(12,5), 
                    gridspec_kw={'height_ratios':[3,1,1]}, 
                    sharex=True)

t_d = t[2*delay:]

ax[0].plot(t,x)
ax[1].scatter(t_d,norms, s=10)

ax[1].set_yscale('log')
ax[1].set_ylim([10**-4,1])

# get locations of anomalies.
anomalies = (norms>=thresh)
where = np.where(anomalies)[0]
where += 2*delay

anom_windowed = np.convolve(anomalies, np.ones(100)/100., mode='same')
ax[2].plot(t_d, anomalies, c='r')
ax2r = ax[2].twinx()

ax2r.plot(t_d, anom_windowed, c='g')

ax[0].scatter(t[where], x[where], c='r', marker='o', s=50, alpha=0.8, zorder=1000)

ax[0].set_title('timeseries (blue) with anomalies (red)', fontsize=16)
ax[1].set_title('normed error in MSET representation', fontsize=16)
ax[2].set_title('anomaly hits (red) and density (green)', fontsize=16)

for axi in ax: axi.xaxis.grid()

fig.tight_layout()

fig.show()
