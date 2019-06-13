run moo.py
X.shape
from mpl_toolkits.mplot3d import Axes3D
fig,ax = pyplot.subplots(1,1)
fig.show()
pyplot.ion()
who
ax.plot(t,x)
anom_windowed
anom_windowed.shape
anomalies.shape
anomalies
ax.scatter(t_d[anomalies],x[(d-1)*delay:][anomalies], c='r', s=10, alpha=0.5)
ax.scatter(t_d[anomalies],x[(d-1)*delay:][anomalies], c='r', s=40, alpha=0.5)
fig2 = pyplot.figure()
import cmocean
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(X[0],X[1],X[2])
ax2.cla()
ax2.scatter(X[0],X[1],X[2], s=40, alpha=0.1)
t_d_n = (t_d-10000)
t_d_n -= t_d_n.min()
t_d_n /= t_d_n.max()
t_d_n.max()
t_d_n = t_d_n / max(t_d_n)
t_d_n.max()
ax2.scatter(X[0],X[1],X[2], c=cmocean.cm.topo(t_d_n), s=30, alpha=0.2)
ax2.cla(); ax2.scatter(X[0],X[1],X[2], c=cmocean.cm.curl(t_d_n), s=30, alpha=0.2)
ax2.cla(); ax2.scatter(X[0],X[1],X[2], c=cmocean.cm.curl(t_d_n), s=60, alpha=0.1)
ax2.cla(); ax2.scatter(X[0],X[1],X[2], c=cmocean.cm.curl(t_d_n), s=100, alpha=0.05)
ax2.cla(); thing = ax2.scatter(X[0],X[1],X[2], c=cmocean.cm.curl(t_d_n), s=100, alpha=0.05)
fig2.colorbar(thing)
fig2.colorbar?
anomalies
ax2.scatter(X[0][delay*(d-1)][anomalies], X[1][delay*(d-1)][anomalies], X[2][delay*(d-1)][anomalies], s=200, c='k', alpha=0.2)
ax2.scatter(X[0][delay*(d-1)][anomalies], X[1][delay*(d-1)][anomalies], X[2][delay*(d-1)][anomalies], s=200, alpha=0.2)
ax2.scatter(X[0][delay*(d-1):][anomalies], X[1][delay*(d-1):][anomalies], X[2][delay*(d-1):][anomalies], s=200, alpha=0.2)
anomalies.shape
X.shape
ax2.scatter(X[0][anomalies], X[1][anomalies], X[2][anomalies], s=200, alpha=0.2)
Out[40].remove()
ax2.scatter(X[0][anomalies], X[1][anomalies], X[2][anomalies], s=200, alpha=0.2, c='k')
Out[40].remove()
Out[42].remove()
ax2.scatter(X[0][anomalies], X[1][anomalies], X[2][anomalies], s=400, alpha=0.5, c='k')
ax2.cla()
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], c=cmocean.cm.curl(t_d_n), s=100, alpha=0.05)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], c=cmocean.cm.cu[:10000]rl(t_d_n), s=100, alpha=0.05)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], c=cmocean.cm.curl(t_d_n)[:10000], s=100, alpha=0.05)
ax2.scatter(X[0][anomalies], X[1][anomalies], X[2][anomalies], s=400, alpha=0.5, c='k')
sum(anomalies)
np.sqrt(276)
1./np.sqrt(276)
Out[50].remove()
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2][:10000], c=cmocean.cm.curl(t_d_n)[:10000], s=100, alpha=0.05)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2][:10000], c=cmocean.cm.curl(t_d_n)[:10000],  alpha=0.05)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2],  alpha=0.05, lw=10)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2],  alpha=0.05, lw=10)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2][:10000],  alpha=0.05, lw=10)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2][:10000],  alpha=0.5, lw=10)
ax2.cla(); thing = ax2.plot(X[0][:10000],X[1][:10000],X[2][:10000], lw=4)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], lw=4)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], lw=4, alpha=0.5)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], lw=4, alpha=0.2)
ax2.cla(); thing = ax2.scatter(X[0][:10000],X[1][:10000],X[2][:10000], lw=4, alpha=0.1)
fig2.axes
fig2.axes[1].remove()
fig2.tight_layout()
fig2.tight_layout()
t_d.shape
import pandas
import glob
ls ../../ma
df = pandas.read_csv('../../ma/CC004_021_wo_RBFnum.csv', header=None)
df
list(df)
df.iloc[:3]
df.columns
fig3,ax3 = pyplot.subplots(4,1,sharex=True)
for i in range(4):
    ax3[i].plot(df.iloc[:,i])
np.diff(df.iloc[:,1])
np.diff(df.iloc[:,0])
df.shape
x.shape
X.shape
ls ../../ma/
fig4,ax4 = pyplot.subplots(4,1)
ax4[0].plot(t,x)
ax4[0].plot(t_d[anomalies],x[(d-1)*delay:][anomalies], c='r', s=40, alpha=0.5)
ax4[0].scatter(t_d[anomalies],x[(d-1)*delay:][anomalies], c='r', s=40, alpha=0.5)
df
ax4[1].plot(df.iloc[:,1]*60,df.iloc[:,0])
niter = df.iloc[:,2].values
niter
anom_ma = np.where(niter>0)
t_ma = df.iloc[:,1].values*60
x_ma = df.iloc[:,0].values
ax4[0].scatter(t_ma[anom_ma], x_ma[anom_ma], c='r', s=40, alpha=0.5)
Out[98].remove()
ax4[0].set_ylim([-2,2])
ax4[0].set_ylim([-3,3])
ax4[0].set_ylim([-3,2])
ax4[0].set_ylim([-2,2])
ax4[1].scatter(t_ma[anom_ma], x_ma[anom_ma], c='r', s=40, alpha=0.5)
ls ../../ma
%paste
ax4[2].plot(t,x)
ax4[1].scatter(t_d[anomalies], x[(d-1)*delay:][anomalies], c='r', s=40, alpha=0.5)
Out[108].remove()
ax4[1].set_ylim([36,40])
ax4[1].set_ylim([35,39])
ax4[2].scatter(t_d[anomalies], x[(d-1)*delay:][anomalies], c='r', s=40, alpha=0.5)
ls ../../ma
df = pandas.read_csv('../../ma/CBA_218_wo_RBFnum.csv')
niter = df.iloc[:,2].values
anom_ma = np.where(niter>0)
t_ma = df.iloc[:,1].values*60
x_ma = df.iloc[:,0].values
ax4[3].scatter(t_ma[anom_ma], x_ma[anom_ma], c='r', s=40, alpha=0.5)
ax4[3].plot(t_ma,x_ma)
sum(anom_ma)
anom_ma
len(anom_ma[0])
for axi in ax4: axi.set_xlim([0,2e4])
for axi in ax4: axi.set_xticks(np.arange(0,20000,1440))
for axi in ax4: axi.set_xticklabels(np.arange(0,20000,1440)//1440)
for axi in ax4: axi.xaxis.grid()
%history -f mset-rbf-comparison-raw.py
ls
import pickle
f = open('cba-218-mset-data.pkl', 'wb')
pickle.dumps?
pickle.dumps({'mouse_id':'CBA-218','delay':6*60,'thresh':0.05,'d':3,'t':t,'x':tempdict[mouse],'mset_t_d':t_d, 'mset_norms':norms, 'mset_anomalies':anomalies},f)
pickle.dump?
pickle.dump({'mouse_id':'CBA-218','delay':6*60,'thresh':0.05,'d':3,'t':t,'x':tempdict[mouse],'mset_t_d':t_d, 'mset_norms':norms, 'mset_anomalies':anomalies},f)
f.close()
%paste
f.close()
ls
f2 = open('cc004-021-mset-data.pkl','wb')
pickle.dump({'mouse_id':'CC004-021','delay':6*60,'thresh':0.05,'d':3,'t':t,'x':tempdict[mouse],'mset_t_d':t_d, 'mset_norms':norms, 'mset_anomalies':anomalies},f2)
f2.close()
mouse
who
fig4.tight_layout()
props = {'facecolor':'w', edgecolor:[0.8,0.8,0.8], alpha=0.8}
props = {'facecolor':'w', 'edgecolor':[0.8,0.8,0.8], 'alpha':0.8}
ax4[0].text(0.95,0.95, 'MSET, CC004-021' ha='right', va='top', fontsize=14, bbox=props)
ax4[0].text(0.95,0.95, 'MSET, CC004-021', ha='right', va='top', fontsize=14, bbox=props)
Out[149].remove()
ax4[0].text(0.95,0.95, 'MSET, CC004-021', ha='right', va='top', fontsize=14, bbox=props, transform=ax4[0].transAxes)
Out[151].remove()
ax4[0].text(0.95,0.9, 'MSET, CC004-021', ha='right', va='top', fontsize=14, bbox=props, transform=ax4[0].transAxes)
Out[153].remove()
ax4[0].text(0.95,0.1, 'MSET, CC004-021', ha='right', va='bottom', fontsize=14, bbox=props, transform=ax4[0].transAxes)
ax4[1].text(0.95,0.1, 'RBF, CC004-021', ha='right', va='bottom', fontsize=14, bbox=props, transform=ax4[1].transAxes)
ax4[2].text(0.95,0.1, 'MSET, CBA-218', ha='right', va='bottom', fontsize=14, bbox=props, transform=ax4[2].transAxes)
ax4[3].text(0.95,0.1, 'RBF, CBA-218', ha='right', va='bottom', fontsize=14, bbox=props, transform=ax4[3].transAxes)
ax4[3].set_xlabel('Time since initial observation (days)', fontsize=14)
ax4[2].set_ylabel('Temperature (degrees Celsius)', fontsize=14)
fig4.tight_layout()
ax4[2].set_ylabel('', fontsize=14)
fig4.tight_layout()
fig4.savefig('mset-rbf-comparison-v1.png', dpi=120, bbox_inches='tight')
fig4.savefig('mset-rbf-comparison-v1.pdf', dpi=120, bbox_inches='tight')
%history -f mset-rbf-comparison-raw.py
