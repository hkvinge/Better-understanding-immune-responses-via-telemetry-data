from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import math
sys.path.append('/data3/darpa/calcom/')
import calcom
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
import scipy
import pywt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

# Use Calcom to load time series
path = '/data4/kvinge/time_series_work/tamu/'
ccd = calcom.io.CCDataSet(path + 'tamu_expts_01-27.h5')

# Time duration for clustering
days = 5
time_duration = days*60*24
compression = 7
clusters = 5
points_to_plot = 131 
# 131 is the max number of time series after processing
pictures_to_check = [] #[42,116,121,112,28,51,90,43,50,33,87,77,47,101,9,55,93,11,81,99]

# Get times of infection
itimes = ccd.get_attr_values('infection_time')
lines = ccd.get_attr_values('line')
liver_cfu = ccd.get_attr_values('liver_cfu')
sacced_early = ccd.get_attr_values('sacced_early')
post_inoc_sac = ccd.get_attr_values('post_inoc_sac')

# Take log of liver cfu values
for count,i in enumerate(liver_cfu):
    if i > 0:
        liver_cfu[count] = math.log10(i)
max_liver_cfu = max(liver_cfu)
for count,i in enumerate(liver_cfu):
    if i > 0:
        liver_cfu[count] = i/max_liver_cfu

# Normalize post_inoc_sac
min_post_inoc = min(post_inoc_sac)
max_post_inoc = max(post_inoc_sac)
post_inoc_sac = [(x-min_post_inoc)/(max_post_inoc - min_post_inoc) for x in post_inoc_sac]

# Grab time seris
ts = []
liver_cfu_corr = []
sacced_early_corr = []
post_inoc_sac_corr = []
ts_added = 0

for count,i in enumerate(ccd.data):
    if (ts_added >= points_to_plot):
        break
    a = utils.process_timeseries(i)
    if len(a) > 0:
        ts_added = ts_added + 1
        a = np.array(a)
        a = a[:,itimes[count]:itimes[count]+time_duration]
        ts.append(a)
        liver_cfu_corr.append(liver_cfu[count])
        sacced_early_corr.append(sacced_early[count])
        post_inoc_sac_corr.append(post_inoc_sac[count])

# Denoise time series using wavelets
ts_thresh = []
for count, i in enumerate(ts):
    if (count > points_to_plot):
    	b_approx = i[0,:]
    for j in range(compression):
    	b_approx, b_detail = pywt.dwt(b_approx, 'db1')
    ts_thresh.append(b_approx)

# Plot some desired time seris
for i in pictures_to_check:
#for i in range(99,130):
    series = ts_thresh[i]
    plt.plot(series)
    plt.title('Time series ' + str(i))
    #plt.show()
    #plt.savefig("ts_" + str(i) + ".png")
    plt.show()

# Convert list of arrrays to matrix
ts_thresh = np.transpose(np.stack(ts_thresh, axis=1))

# Get the dimensions of the final time series
dims = np.shape(ts_thresh)

# Do a cyclic permutation of a healthy mouse. 
generic_values = ts_thresh[3,:]
generic_values = np.reshape(generic_values,(1,dims[1]))
for j in range(dims[1]):
    b = np.roll(generic_values[0,:],j)
    b = np.reshape(b,(1,dims[1]))
    generic_values = np.concatenate((generic_values,b),axis=0)

for i in range(dims[1]):
    plt.plot(generic_values[i,:])
plt.show()

orbit_size = np.shape(generic_values)

#ts_thresh = np.concatenate((ts_thresh,generic_values),axis=0)

distance_matrix = np.zeros((dims[0],dims[0]))

# Generate the distance matrix
#distance_matrix = scipy.spatial.distance.pdist(ts_thresh,'correlation')
#distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
for i in range(dims[0]):
    print(i)
    for j in range(dims[0]):
        a = scipy.spatial.distance.correlation(ts_thresh[i,:],ts_thresh[j,:])
        c = a
        b = a
        for k in range(orbit_size[0]):
            temp = scipy.spatial.distance.correlation(ts_thresh[i,:],generic_values[k,:])
            if (temp < c):
                c = temp
        for k in range(orbit_size[0]):
            temp = scipy.spatial.distance.correlation(ts_thresh[j,:],generic_values[k,:])
            if (temp < b):
                b = temp
        
        distance_matrix[i,j] = min(a,b+c)

# Get size of distance matrix
dims = np.shape(distance_matrix)
	
# Run MDS on distance matrix
# Alter distance matrix
M = -.5*np.square(distance_matrix)

# Create mean centering matrix
H = np.identity(dims[0]) - (1/points_to_plot)*np.ones((dims[0],dims[0]))
B = np.matmul(M,H)
B = np.matmul(H,B)

# Take singular value decomposition
U,S,Vh = scipy.linalg.svd(B)
Sroot = np.sqrt(S)

# Make singular value vector into diagonal matrix
Sroot = np.diag(Sroot)

# Compute embedding
embedding = np.matmul(U,Sroot)

# Save the matrix as a CSV file
np.savetxt("distance_matrix.csv",embedding,delimiter=",")

# Run k-means on the embedding
kmeans = KMeans(n_clusters = clusters, random_state=0).fit(embedding)
labels = kmeans.labels_


# Create dictionary for colors
color_dict = {0:'red',1:'blue',2:'green',3:"cyan",4:"magenta",5:"yellow",6:"black",7:"black",8:"white"}

# Plot points with K-means labels
for i in range(dims[0]):
    plt.plot([embedding[i,0]],[embedding[i,1]],marker='o',color=color_dict[labels[i]])
for i in range(dims[0]):
    plt.annotate(i,(embedding[i,0],embedding[i,1]))
plt.show()

# Plot points with K-means labels
for i in range(dims[0]):
    plt.plot([embedding[i,0]],[embedding[i,1]],marker='o',color=color_dict[labels[i]])
plt.show()

# Plot points with K-means labels
#for i in range(dims[0]):
plt.scatter(embedding[:,0],embedding[:,1],marker='o',c=embedding[:,2],cmap="cool")
plt.show()

# Plot the embedding
fig = plt.figure()
ax = Axes3D(fig)
for i in range(dims[0]):
    ax.scatter(embedding[i,0], embedding[i,1], embedding[i,2], c =color_dict[labels[i]], marker='o')
plt.show()

# Choose colormap for continuous values
blues = plt.get_cmap('bwr')

# Plot with respect to liver CFU
for i in range(dims[0]):
    if liver_cfu_corr[i] >= 0:
        plt.scatter(embedding[i,0],embedding[i,1],marker='o',c=blues(liver_cfu_corr[i]))
plt.show()

# Plot with respect to post inoculation sacrifice
for i in range(dims[0]):
    plt.scatter(embedding[i,0],embedding[i,1],marker='o',c=blues(post_inoc_sac_corr[i]))
plt.show()
