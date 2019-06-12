import numpy as np
from matplotlib import pyplot
try:
    import cmocean
    mycm = cmocean.cm.dense
except:
    mycm = pyplot.cm.viridis
#

x0 = [0.5,1]
nc = 21 # number of contour levels

X,Y = np.meshgrid( np.linspace(-4,4,1601), np.linspace(-4,4,1601))
Z =  1. - (np.sqrt((X-x0[0])**2 + (Y-x0[1])**2))/(np.linalg.norm(x0) + np.sqrt(X**2 + Y**2))


fig,ax = pyplot.subplots(1,1)


cax = ax.contourf(X,Y,Z,nc, cmap=mycm)

# solution to existence of edge lines in pdf rendering.
# See https://stackoverflow.com/a/32911755
for c in cax.collections:
    c.set_edgecolor("face")

ax.axis('square')
cbar = fig.colorbar(cax)
cbar.set_ticks(np.arange(0,1.1,0.2))

#ax.contour(X,Y,Z,nc, colors=[0.2*np.ones(3)], linewidths=0.2, alpha=0.2)

ax.grid(lw=0.5, c='k', alpha=0.1)
fig.tight_layout()
fig.savefig('vector-otimes-vis-2d.png', dpi=120, bbox_inches='tight')
fig.savefig('vector-otimes-vis-2d.pdf', dpi=120, bbox_inches='tight')

#fig.show()
