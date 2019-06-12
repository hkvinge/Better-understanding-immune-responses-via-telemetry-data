import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.sparse import linalg
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def lap_eig_sparse(data, technique = 'eps_ball', k = 5, weights = 'simple'):
    """
    Applies Laplacian Eigenmap to (LARGE) m by n data matrix with m features and n samples.
    Returns eigenvectors and eigenvalues (use eigenvectors 1 and 2 for 2D embedding)
    Inputs:
        technique - {'knn', 'eps_ball'}
        weights - {'heat_kernel', 'simple'}

    Outputs:
        eigenvalues, eigenvectors
    """
    if technique == "eps_ball":
        num_col = data.shape[1]
        Distance = np.empty((num_col, num_col))   #init distance matrix
        for i in range(num_col):
            for j in range(num_col):
                Distance[(i,j)] = np.linalg.norm(data[:,i] - data[:,j])

        epsilon = np.mean(Distance)

        A = csr_matrix((num_col, num_col))
        for i in range(num_col):
            for j in range(num_col):
                if Distance[(i,j)] < epsilon:
                    if weights == 'simple':
                        A[i,j] = 1
                    elif weights == 'heat_kernel':
                        A[i,j] = np.exp(-(Distance[i,j]**2)/1)
                else:
                    continue

    elif technique == "knn":
        data = np.transpose(data)
        nn = NearestNeighbors()
        nn.fit(data)
        if weights == 'simple':
            A = nn.kneighbors_graph(n_neighbors = k, mode = 'connectivity')
        elif weights == 'heat_kernel':
            A = nn.kneighbors_graph(n_neighbors = k, mode = 'distance')
            A = (-A**2)/1
            A[A == 0] = -np.inf
            A = np.exp(A)
        #make A symmetric
        #they are not always positive, though, which makes for bad embedding
        B = np.nonzero(A)
        Anew = A + A.T
        Anew[B] = A[B]
        A = Anew

    # init weight matrix D which has diagonal entries as sum of each row of W
    D = diags(np.sum(A, axis = 1).T.tolist()[0])

    Laplacian = D - A

    # solve generalized eigenvector problem
    evals, evecs = linalg.eigs(Laplacian, k = 4, M = D, which = 'SM')

    return evals, evecs

def lap_eig(data, technique = 'eps_ball', k = 5, weights = 'simple'):
    """
    Applies Laplacian Eigenmap to m by n data matrix with m features and n samples.
    Returns eigenvectors and eigenvalues (use eigenvectors 1 and 2 for 2D embedding)
    Inputs:
        technique - {'knn', 'eps_ball'}
        weights - {'heat_kernel', 'simple'}

    Outputs:
        eigenvalues, eigenvectors
    """
    if technique == "eps_ball":
        num_col = data.shape[1]
        Distance = np.empty((num_col, num_col))   #init distance matrix
        for i in range(num_col):
            for j in range(num_col):
                Distance[(i,j)] = np.linalg.norm(data[:,i] - data[:,j])

        epsilon = np.mean(Distance)

        A = np.zeros((num_col, num_col))
        for i in range(num_col):
            for j in range(num_col):
                if Distance[(i,j)] < epsilon:
                    if weights == 'simple':
                        A[i,j] = 1
                    elif weights == 'heat_kernel':
                        A[i,j] = np.exp(-(Distance[i,j]**2)/1)
                else:
                    continue

    elif technique == "knn":
        data = np.transpose(data)
        nn = NearestNeighbors()
        nn.fit(data)
        if weights == 'simple':
            A = nn.kneighbors_graph(n_neighbors = k, mode = 'connectivity')
            A = A.toarray()
        elif weights == 'heat_kernel':
            A = nn.kneighbors_graph(n_neighbors = k, mode = 'distance')
            A = A.toarray()
            A = (-A**2)/1
            A[A == 0] = -np.inf
            A = np.exp(A)
        #make A symmetric to guarantee real eigenvalues
        #they are not always positive, though, which makes for bad embedding
        B = np.nonzero(A)
        Anew = A + A.T
        Anew[B] = A[B]
        A = Anew

    # init weight matrix D which has diagonal entries as sum of each row of W
    D = np.diag(np.sum(A, axis = 1))

    Laplacian = D - A

    # solve generalized eigenvector problem
    evals, evecs = scipy.linalg.eig(Laplacian,D)

    # sort evals and evecs
    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    #plot times for increasing data matrix size
    dim = 100
    max_samples = 1000
    x = np.arange(10, max_samples, 10)
    duration = []
    for i in range(10,max_samples,10):
        matrix = np.random.random((dim,i))
        start = time.time()
        evals, evecs = lap_eig(matrix, technique = 'knn', k = 5, weights = 'simple')
        end = time.time()
        duration.append(end - start)
        print("sample size: " + str(i))
    fig, ax = plt.subplots()
    z = np.polyfit(x, duration, deg = 2)
    fit = z[2] + z[1]*x + z[0]*x**2
    ax.plot(x,fit)
    ax.scatter(np.arange(10, max_samples, 10), duration)
    plt.show(block = False)
    print('Estimated duration for sample size = 12000: ' + str(z[2] + z[1]*12000 + z[0]*12000**2))
