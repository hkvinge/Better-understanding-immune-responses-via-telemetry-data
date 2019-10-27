'''
A script to reproduce laplacian eigenmaps on mice timeseries data

V2 - COMBO FIGURE WOW
'''
from LapEig_mice_utils import MyMice as MM
import scipy
import numpy as np
from laplacian_eigenmap import lap_eig
import matplotlib.pyplot as plt
import sys
#sys.path.append('C://Users/Pat R/Documents/Github/calcom')
import calcom

# wow
import matplotlib.gridspec as gridspec


#file_path = 'C://Users/Pat R/Documents/Github/Discovering-Signatures-in-Telemetry-Data/tamu_expts_01-28.h5'
file_path = '/data3/darpa/tamu/tamu_expts_01-28.h5'


window = 4320 # this will likely never change.
num_windows = 12
win_len = 1440 #change to 2880 for new experiment
mouses = [206, 127, 70, 176, 191, 3, 6] # list of mice indices to label in final embedding

# gather and process data
ccd = calcom.io.CCDataSet(file_path)

####################################################

mice = MM(ccd)
mice.get_windowed_data(window, process_data = True)

## one-day window experiment with lap_eig
mice.get_sliding_windows(win_len, n = num_windows, normalize = True, days = 6, type = 'mean')

sliding_eigenvals = np.empty((num_windows - 1, mice.data.shape[1], mice.data.shape[1]))
sliding_eigenvecs = np.empty((num_windows - 1, mice.data.shape[1], mice.data.shape[1]))
for i in range(num_windows - 1):
    sliding_eigenvals[i,:,:], sliding_eigenvecs[i,:,:] = lap_eig(mice.sliding_window[i,:,:], technique = 'knn', k = 5, weights = 'simple')
    if sliding_eigenvecs[i,0,1] < 0:
        sliding_eigenvecs[i,:,1] = -sliding_eigenvecs[i,:,1]
    if sliding_eigenvecs[i,0,2] < 0:
        sliding_eigenvecs[i,:,2] = -sliding_eigenvecs[i,:,2]
    if sliding_eigenvecs[i,0,3] < 0:
        sliding_eigenvecs[i,:,3] = -sliding_eigenvecs[i,:,3]

#########################################################

# plot results

#########################################################

#plot_multiple_embeddings(mice, sliding_eigenvals, sliding_eigenvecs, num_windows, win_len, mouses, save = False)

evecs = sliding_eigenvecs



fig = plt.figure(constrained_layout=False, figsize=(16,8))
gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.0, width_ratios=[0.3,0.7])
gss = []
gss.append( gs[0].subgridspec(3,1, wspace=0.0, hspace=0.05) )
gss.append( gs[1].subgridspec(7,1, wspace=0.0, hspace=0.05) )

ax = []
for j,wl in enumerate( [3,7] ):
    ax.append([])
    for i in range(wl):
        if i==0:
            ax[j].append( fig.add_subplot( gss[j][i,0] ) )
        else:
            if j==1:
                ax[j].append( fig.add_subplot( gss[j][i,0] , sharex=ax[j][0], sharey=ax[j][0]) )
            else:
                ax[j].append( fig.add_subplot( gss[j][i,0] , sharex=ax[j][0]) )
        if i!=wl-1:
            ax[j][i].set_xticklabels([])
#
for i in range(3):
    ax[0][i].set_xticks([])
    ax[0][i].set_yticks([])
#
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
fig.show()


#plot_multiple_embeddings(mice, evals, evecs, num_windows, win_len, mouses, save = False)
graph_idx = [0,6,10]
#fig, ax = plt.subplots(3, 1, figsize = (6,15))
#fig.suptitle('2-D Embeddings of One-Day Windows \n of Temperature Time Series', weight = 'bold')
for i, idx in enumerate(graph_idx):
    start_of_window = (6/num_windows) * idx - 3
#    if start_of_window < 0:
#        ax[i].set_title('Window starts {} days pre infection, X[-3 day, -2 day]'.format(np.abs(start_of_window)))
#    elif start_of_window == 0:
#        ax[i].set_title('Window starts at infection, X[0, 1 day]')
#    else:
#        ax[i].set_title('Window starts {} days post infection, X[2 day, 3 day]'.format(start_of_window))

    ax[0][i].scatter(evecs[idx,:,1], evecs[idx,:,2], s = 10)
    ax[0][i].get_xaxis().set_visible(False)
    ax[0][i].get_yaxis().set_visible(False)

    for j, text in enumerate(mice.mouse_id_list):
        if j in mouses:
            ax[0][i].scatter(evecs[idx,j,1], evecs[idx,j,2], s = 12, color = 'red')
            # one-off tweak blah
            if i!=2:
                ax[0][i].annotate(text, (evecs[idx,j,1], evecs[idx,j,2]), weight = 'bold', fontsize=13)
            else:
                width = np.diff( ax[0][0].get_xlim() )[0]
                ax[0][i].annotate(text, (evecs[idx,j,1], evecs[idx,j,2]), weight = 'bold', xytext=(evecs[idx,j,1]-0.20*width,evecs[idx,j,2]), textcoords='data', fontsize=13)
        else:
            ax[0][i].scatter(evecs[idx,j,1], evecs[idx,j,2], s = 10, color = 'lightblue', alpha = 0.2)

#########################################################

#this bit plots the 7 mice timeseries that we care about along with highlighted windows
x = np.arange(2*window) - window
#fig, ax = plt.subplots(7,1, figsize = (10,25)) #might need to play with figsize
for j, idx in enumerate(mouses):
    ax[1][j].cla()
    ax[1][j].plot(x, mice.data[:,idx], label = mice.mouse_id_list[idx])
    ax[1][j].axvline(x = 0, c = 'red')
    ax[1][j].axvspan(-window, win_len - window, color = 'green', alpha = 0.2)
    ax[1][j].axvspan(6*win_len/2 - window, (6/2+1)*win_len - window, color = 'yellow', alpha = 0.2)
    ax[1][j].axvspan(10*win_len/2 - window, (10/2+1)*win_len - window, color = 'red', alpha = 0.2)
    ax[1][j].legend(loc = 'lower left', fontsize=14)
    
    ax[1][j].set_xticks(np.arange(-1440*3,1440*3+1,1440))
    ax[1][j].grid()
    
#    if j != 6:
#        ax[1][j].get_xaxis().set_visible(False)
#plt.show()
#

ax[1][0].set_yticks(np.arange(34,39,2))
ax[1][-1].set_xticklabels(np.arange(-3,4), fontsize=14)
for j in range(7):
    ax[1][j].set_yticklabels(np.arange(34,39,2), fontsize=14)
#

# more labeling!!!!!!
ax[1][-1].set_xlabel('Time (days)', fontsize=16)
for j,color,interval in zip(range(3), ['g','y','r'], [r'$[-3,-2]$', r'$[0,1]$', r'$[2,3]$']):
#    xint = ax[0][j].get_xlim()
    yint = ax[0][j].get_ylim()
#    ax[0][j].set_xlim([xint[0], xint[0] + 1.2*np.diff(xint)])
    ax[0][j].set_ylim([yint[0], yint[0] + 1.2*np.diff(yint)])
    
    ax[0][j].text(0.95,0.95, 'Embedding time: '+interval, fontsize=16, ha='right', va='top', transform=ax[0][j].transAxes, bbox=dict(facecolor=color, alpha=0.1))
#

ax[1][3].set_ylabel(r'Temperature ($^{\circ}$C)', fontsize=16)

fig.show()
fig.savefig('lap_eig_combo_vis_generator.png', dpi=120, bbox_inches='tight')
