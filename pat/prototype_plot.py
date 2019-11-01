import scipy
import numpy as np
import matplotlib.pyplot as plt
import calcom

from LapEig_mice_utils import MyMice as MM

#file_path = 'C://Users//patjr//Documents//code//datasets//tamu_expts_01-28.h5'
file_path = '/data3/darpa/tamu/tamu_expts_01-28.h5'


#################################

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

#################

# (FIGURE 1) plot prototypical mice
x = (np.arange(2*window) - window) * (1/1440)
fig, ax = plt.subplots(5, 1, figsize = (12,8), sharex=True, sharey=True)

for i, _id in enumerate(proto_mice_list):
    ax[i].plot(x, mice.data[:, proto_mice_idx[i]])
    ax[i].axvline(x = 0, c = 'red')
ax[-1].set_xlabel('Time (days)', fontsize = 16)


ax[-1].set_xticklabels( np.array(ax[-1].get_xticks(), dtype=int) , fontsize=14)

ax[-1].set_ylim([34,39])

for axi in ax: axi.set_yticklabels( np.array(ax[-1].get_yticks(), dtype=int) , fontsize=14)
for axi in ax: axi.grid()

fig.tight_layout()
# ylabel screws with axis spacing in tight layout
ax[2].set_ylabel(r'Temperature ($^{\circ}$C)', fontsize = 16)

fig.savefig('prototype_mice_vis.png', dpi=120, bbox_inches='tight')
fig.savefig('prototype_mice_vis.pdf', dpi=120, bbox_inches='tight')
