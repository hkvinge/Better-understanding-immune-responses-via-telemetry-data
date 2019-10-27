def mds(D):
    '''
    Implementation of Multidimensional Scaling -- also known as 
    principal coordinates analysis.
    
    Inputs:
        D - dissimilarity matrix. Only real requirement is that D is symmetric.
    Outputs:
        embedding - n-by-2 of coordinates for the MDS embedding.
    
    This version does the full eigendecomposition of the associated 
    matrix. Will be very slow for large D.
    '''
    import numpy as np
    
    n,_ = np.shape(D)
    e = np.ones( (n,1) )
    
    H = np.eye( n,n ) - 1./n * np.dot( e, e.T )
    
    L = -0.5*np.dot(H, np.dot(D,H) )
    
    w,v = np.linalg.eig(L) # eigenvalues, vectors
    
    order = np.argsort( -w )
    
    if any(w[order[:2]]<0.):
        print('Warning: at least one of the two largest eigenvalues is negative.')
    #
    embedding = v[:,order[:2]] * np.sqrt( np.abs(w[order[:2]]) )
    return embedding
#

if __name__ == "__main__":
    import numpy as np
    import metrics
    from matplotlib import pyplot
    
    # Generate sinusoids with random phases and some noise.
    N = 100
    eps = 1
    phases = 2*np.pi* np.random.rand(N)
    
    t = np.linspace(0,4*np.pi, 500)
    xs = [ eps*np.random.randn(len(t)) + np.sin(t-p) for p in phases ]
    
    D = np.zeros( (N,N) )
    for j in range(N):
        for i in range(j):
            # note: if you try d_qc here, it will take much longer.
            D[i,j] = metrics.d_c(xs[i], xs[j])
            D[j,i] = D[i,j]
    #
    embedding = mds(D)
    
    fig,ax = pyplot.subplots(1,1, figsize=(8,6))
    cax = ax.scatter(embedding[:,0], embedding[:,1], c=phases, vmin=0, vmax=2*np.pi, cmap=pyplot.cm.twilight)
    
    cbar = fig.colorbar(cax)
    cbar.set_ticks( np.arange(0,2*np.pi+0.1, np.pi/2) )
    cbar.set_ticklabels([r'$\frac{%i\pi}{2}$'%j for j in range(5)])
    cbar.ax.tick_params(labelsize=14)   #ugh
    
    fig.suptitle('Embedding of sinusoids via MDS; coloration by phase', fontsize=16)

    fig.show()
    pyplot.ion()
#
