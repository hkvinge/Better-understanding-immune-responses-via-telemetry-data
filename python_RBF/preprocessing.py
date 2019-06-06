import numpy as np
import pdb

class PreProcessing:
    def __init__(self, data, time_idx, smooth_time, embedding_dimension, predict_time):
        self.raw_data = data
        self.time_index = time_idx
        self.lag = smooth_time
        self.dim = embedding_dimension
        self.predict_time = predict_time

    def clean(self):
        data = self.raw_data[~np.isnan(self.raw_data)]
        index = self.time_index[~np.isnan(self.time_index)]
        return data, index

    def smooth(self, data, index):
        n = self.lag
        smoothed_data = np.convolve(data, np.ones(n)/n, mode='valid')
        smoothed_index = index[self.lag-1:]
        return smoothed_data, smoothed_index
    'TODO:dubug autocorrelation function'
    def autocorrelation(self, data):
        mean_data = data - np.mean(data)
        correlation = np.correlate(mean_data, mean_data, mode='full')
        acf = correlation[int(np.floor(correlation.size/2)):]
        idx = 0
        for i in acf:
            if i < 0:
                break
            else:
                idx = idx + 1
        return idx

    def time_delayed_embed(self, data, delay, index):
        dim = self.dim
        length = len(data) - self.predict_time - (dim-1) * delay
        input_data = np.zeros([dim, length])
        output_data = np.zeros([length])
        index_new = np.zeros([length])
        #pdb.set_trace()
        for i in range(0, length):
            idx = np.array([i, i+1, i+2])
            input_data[:, i] = data[idx]
            current_idx = i+(dim-1)*delay+self.predict_time
            output_data[i] = data[current_idx]
            index_new[i] = index[current_idx]
        return input_data, output_data, index_new


