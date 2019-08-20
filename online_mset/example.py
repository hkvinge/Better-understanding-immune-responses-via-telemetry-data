import mset

def visualize_mset(x,thresh,delay):
    import tde
    import numpy as np
    from matplotlib import pyplot

    # time delayed embedding of the scalar time series based on input delay.
    X = tde.tde(x, delay=delay)

    # MSET algorithm on the vector time series, with print statements.
    norms = mset.online_mset(X, output_norms=True, thresh=thresh, verbosity=1)

    # visualize
    fig,ax = pyplot.subplots(3,1, 
                        figsize=(12,5), 
                        gridspec_kw={'height_ratios':[3,1,1]}, 
                        sharex=True)

    t_d = t[2*delay:]

    ax[0].plot(t,x)
    ax[1].scatter(t_d,norms, s=10)

    ymin = 10**int(np.floor(min(np.log10(norms[norms!=0.]))))
    ymax = 10**int(np.ceil(max(np.log10(norms[norms!=0.]))))

    ax[1].set_yscale('log')
    ax[1].set_ylim([ymin,ymax])
    yticks = [ymin,thresh,ymax]
    ax[1].set_yticks( yticks )
    ax[1].set_yticklabels([r'$10^{%i}$'%np.log10(val) for val in yticks])
    
    
    ax[1].yaxis.grid()
    gls = ax[1].get_ygridlines()
    gls[1].set_color('r')

    # get locations of anomalies.
    anomalies = (norms>=thresh)
    where = np.where(anomalies)[0]
    where += 2*delay

    anom_windowed = np.convolve(anomalies, np.ones(delay//2)/(delay/2.), mode='same')
    ax[2].plot(t_d, anomalies, c='r')
    ax2r = ax[2].twinx()

    ax2r.plot(t_d, anom_windowed, c='g')

    ax[0].scatter(t[where], x[where], c='r', marker='o', s=50, alpha=0.8, zorder=1000)

    ax[0].set_title('timeseries (blue) with anomalies (red)', fontsize=16)
    ax[1].set_title('normed error in MSET representation', fontsize=16)
    ax[2].set_title('anomaly hits (red) and density (green)', fontsize=16)

    for axi in ax: axi.xaxis.grid()

    fig.tight_layout()

    return fig,ax
#

if __name__=="__main__":
    '''
    Example application of MSET to a scalar time series 
    by applying a time delayed embedding. The time series used 
    here is a noisy sinusoid with a transition in its period 
    halfway through the observation period.
    '''
    import numpy as np
    
    n = 10000
    thresh = 0.1

    t = np.linspace(0, 10*np.pi, n)
    x = np.sin(t) + 0.1*np.random.randn(n)

    # introduce a new type of signal halfway through.
    heavy = (1 + np.tanh( 0.5*(t-5*np.pi) ))/2.
    x += heavy*np.sin(t/2) + heavy*(-np.sin(t))

    # time delayed embedding based on analytical 
    # zero-autocorrelation time of pi/2 for a sinusoid.
    delay = int((np.pi/2)/(t[1] - t[0]))
    
    fig,ax = visualize_mset(x,thresh,delay)
    fig.show()

#    fig.savefig('online_tde_mset.png', dpi=120, bbox_inches='tight')
#
