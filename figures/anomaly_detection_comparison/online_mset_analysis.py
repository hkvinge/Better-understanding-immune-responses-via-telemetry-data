import sys

PREFIX = '/home/katrina/a/aminian/'
repo = 'Better-understanding-immune-responses-via-telemetry-data/'

sys.path.append(PREFIX + repo)
sys.path.append(PREFIX + repo + 'online_mset')

import tde
import mset
import load_csvs

tdict,df = load_csvs.load()

mice = ['CC004-021', 'CBA-218']

# wow, parameters
thresh = 0.05
delay = 6*60    # six hours

figs = []
axs = []


for m in mice:
    # cheat some - mean-center the data with
    # some amount of pre-infection data.
    temp = tdict[m]
    temp -= sum(temp[:4*delay])/(4.*delay)
    
    fig,ax = mset.visualize_mset(temp,thresh,delay)
    figs.append(fig)
    axs.append(ax)
    fig.show()
#
