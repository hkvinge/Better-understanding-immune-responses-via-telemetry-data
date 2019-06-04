# Better understanding of immune responses via telemetry data
This is the repository associated with a paper that explains how mouse temperature data 
can be used to better understand and predict reactions to infection.


## Requirements
Packages:

* numpy
* scipy
* matplotlib
* calcom (dependency may be removed later)
* PyWavelets (via, for example, `pip install PyWavelets --user` or similar pip command)

Files:

* `tamu_expts_01-27.h5`

## Usage
The file `utils.py` contains a large number of utilities related to processing 
the mice timeseries in the experiment, and also handles some preprocessing. 
If you are running this script outside of Katrina, then you need to 
figure out the folder containing `tamu_expts_01-27.h5` and add it to the 
list `prefixes` in `utils.py`. This is simply a list of potential locations 
for the file that the script searches.

The file `wavelet_smoothing.py` applies smoothing to the raw timeseries 
using Daubechies db1 wavelets and outputs the smoothed *temperature* timeseries 
only to the variable `ts_smooth`. Typically you should use this functionality 
via `import wavelet_smooth as ws` (for example) then access the data via 
`ws.ts_smooth[0]`, `ws.ts_smooth[1]`, etc.