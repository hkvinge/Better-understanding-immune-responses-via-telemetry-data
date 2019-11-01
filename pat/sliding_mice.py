'''
A script to reproduce laplacian eigenmaps on mice timeseries data
'''
from LapEig_mice_utils import MyMice as MM
import scipy
import numpy as np
from laplacian_eigenmap import lap_eig
import matplotlib.pyplot as plt
import calcom
# file_path = 'C://Users/Pat R/Documents/Github/Discovering-Signatures-in-Telemetry-Data/tamu_expts_01-28.h5'
file_path = 'C://Users//patjr//Documents//code//datasets//tamu_expts_01-28.h5'

def plot_multiple_embeddings(mice, evals, evecs, num_windows, win_len, mouses, save = False):
    '''
    plots 3 lap_eig embeddings in one figure
    '''
    graph_idx = [0,6,10]
    fig, ax = plt.subplots(3, 1, figsize = (6,15))
    fig.suptitle('2-D Embeddings of One-Day Windows \n of Temperature Time Series', weight = 'bold')
    for i, idx in enumerate(graph_idx):
        start_of_window = (6/num_windows) * idx - 3
        if start_of_window < 0:
            ax[i].set_title('Window starts {} days pre infection'.format(np.abs(start_of_window)))
        elif start_of_window == 0:
            ax[i].set_title('Window starts at infection')
        else:
            ax[i].set_title('Window starts {} days post infection'.format(start_of_window))
        ax[i].scatter(evecs[idx,:,1], evecs[idx,:,2], s = 10)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

        for j, text in enumerate(mice.mouse_id_list):
            if j in mouses:
                ax[i].scatter(evecs[idx,j,1], evecs[idx,j,2], s = 12, color = 'red')
                ax[i].annotate(text, (evecs[idx,j,1], evecs[idx,j,2]), weight = 'bold')
            else:
                ax[i].scatter(evecs[idx,j,1], evecs[idx,j,2], s = 10, color = 'lightblue', alpha = 0.2)

    plt.show()
    if save:
        fig.savefig('{:03d}'.format(i))

def plot_one_embedding(mice, evals, evecs, num_windows, win_len, mouses, save = False):
    '''
    plots one lap_eig embedding figure at a time
    '''
    graph_idx = [0,6,10]
    for i, idx in enumerate(graph_idx):
        fig, ax = plt.subplots()
        start_of_window = (6/num_windows) * idx - 3
        if start_of_window < 0:
            ax.set_title('Window starts {} days pre infection'.format(np.abs(start_of_window)))
        elif start_of_window == 0:
            ax.set_title('Window starts at infection')
        else:
            ax.set_title('Window starts {} days post infection'.format(start_of_window))
        ax.scatter(evecs[idx,:,1], evecs[idx,:,2], s = 10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j, text in enumerate(mice.mouse_id_list):
            if j in mouses:
                ax.scatter(evecs[idx,j,1], evecs[idx,j,2], s = 12, color = 'red')
                ax.annotate(text, (evecs[idx,j,1], evecs[idx,j,2]), weight = 'bold')
            else:
                ax.scatter(evecs[idx,j,1], evecs[idx,j,2], s = 10, color = 'lightblue', alpha = 0.2)

        plt.show()
        if save:
            fig.savefig('{:03d}'.format(i))


# init params
window = 4320 # this will likely never change.
num_windows = 12
win_len = 1440 #change to 2880 for new experiment
mouses = [206, 127, 70, 176, 191, 3, 6] # list of mice indices to label in final embedding

# gather and process data
ccd = calcom.io.CCDataSet(file_path)


mice = MM(ccd)
mice.get_windowed_data(window, process_data = True)

proto_mice_list = ['CC023-183', 'CC011-306', 'CC023-192', 'CBA-119', 'CC042-080']
proto_mice_idx = [133, 76, 137, 15, 203]

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


# # plot results
# plot_multiple_embeddings(mice, sliding_eigenvals, sliding_eigenvecs, num_windows, win_len, mouses, save = False)
# plot_one_embedding(mice, sliding_eigenvals, sliding_eigenvecs, num_windows, win_len, mouses, save = False)


# # (FIGURE 1) plot prototypical mice
# x = (np.arange(2*window) - window) * (1/1440)
# fig, ax = plt.subplots(5, 1, figsize = (10,25))
# for i, _id in enumerate(proto_mice_list):
#     ax[i].plot(x, mice.data[:, proto_mice_idx[i]])
#     ax[i].axvline(x = 0, c = 'red')
#
# ax[-1].set_xlabel('Time (days)', fontsize = 16)
# ax[2].set_ylabel(r'Temperature ($^{\circ}$C)', fontsize = 16)
# fig.suptitle('Prototypical Signatures in Mice Temperature Time Series', fontsize = 16)
# plt.show()


# (FIGURE 4) plot the 7 mice timeseries that we care about along with highlighted windows
x = (np.arange(2*window) - window) * (1/1440)
fig, ax = plt.subplots(7,1, figsize = (10,25)) #might need to play with figsize
for j, idx in enumerate(mouses):
    ax[j].cla()
    ax[j].plot(x, mice.data[:,idx], label = mice.mouse_id_list[idx])
    ax[j].axvline(x = 0, c = 'red')
    ax[j].axvspan((-window) * (1/1440), (win_len - window) * (1/1440), color = 'green', alpha = 0.2)
    ax[j].axvspan((6*win_len/2 - window) * (1/1440), ((6/2+1)*win_len - window) * (1/1440), color = 'yellow', alpha = 0.2)
    ax[j].axvspan((10*win_len/2 - window) * (1/1440), ((10/2+1)*win_len - window) * (1/1440), color = 'red', alpha = 0.2)
    ax[j].legend(loc = 0)
    if j != 6:
        ax[j].get_xaxis().set_visible(False)
ax[-1].set_xlabel('Time (days)', fontsize = 16)
ax[3].set_ylabel(r'Temperature ($^{\circ}$C)', fontsize = 16)
plt.show()
