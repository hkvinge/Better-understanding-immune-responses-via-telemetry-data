Sequential RBF for online construction of a model 
to fit to time series data. The version here is designed 
for a scalar timeseries. Time-delayed-embedding sends it to 
three dimensions (by default) with a lag time chosen 
either "arbitrarily" or based on the first 
zero-autocorrelation time of the signal.

The file SequentialRBF.py demonstrates the basic usage 
on a noisy sinusoidal time series.

See

Ma, X; Aminian, M; Kirby, M; "Error-adaptive modeling of streaming 
time-series data using radial basis functions"; Journal of 
Computational and Applied Mathematics; Available online 19 November 2018 
for algorithmic details and analysis.
