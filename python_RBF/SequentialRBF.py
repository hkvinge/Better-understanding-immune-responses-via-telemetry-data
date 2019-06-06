import numpy as np
try:
    from .LinearProgramming import LinearProg
except:
    from LinearProgramming import LinearProg
#
try:
    from .preprocessing import *
except:
    from preprocessing import *
#


class SequentialRbf:
    def __init__(self, number_center=500):

        self.n = number_center + 1
        self.y = None
        self.t = None
        self.actual_t = None
        self.current_val = None
        self.old_lp = None
        self.new_lp = None
        'key parameter: C_const is a sensitivity parameter'
        self.C_const = 1.
        self.EPS = 1e-10
        self.DerivativeCutOff = 1e-5

        'RBF funciton'
        self.phi = None
        'Shape function'
        self.z = None
        'RBF parameters'
        self.centers = []

        'embedding parameters'
        # self.embed_dim = None
        # self.predict_time = None
        # self.smooth_time = None
        self.embed_dim = 3
        self.smooth_time = 0
        self.predict_time = 1

        'output values'
        self.eps_timeseries = []
        self.nrbf_timeseries = []
        self.niter = []

        'Level of prints'
        self.verbosity = 0
    #

    def set_functions(self, rbf_function, shape_function):
        self.phi = rbf_function
        self.z = shape_function

    def set_c(self, c):
        self.phi.c = c
        self.z.c = c

    def get_phi_times_z(self):
        return self.phi.get_val()*self.z.get_val()

    def generate_centers(self):
        #centers = np.zeros(self.embed_dim, self.n-1)
        y_max = np.max(self.y)
        y_min = np.min(self.y)
        centers = y_min + (y_max-y_min)*np.random.rand(self.embed_dim, self.n-1)
        return centers

    def build_initial_lp(self):

        n = self.n
        en = np.ones([n, 1])
        e2n = np.ones([2*n, 1])
        En = np.eye(n)
        E2n = np.eye(2*n)
        'definition of initial A'
        M1 = En
        M2 = -1*En
        M3 = np.hstack((M1, M2))
        M4 = np.hstack((M2, M2))
        A = np.vstack((M3, M4))
        A_neg = -1*A
        A_aug = np.hstack((A, A_neg))
        A_col = np.zeros([2*n,1])
        A_mod = np.hstack((A_aug, A_col))
        A_initial = np.hstack((A_mod, E2n))
        'definition of initial b'
        b_initial = np.zeros([2*n, 1])
        'definition of initial c'
        c1 = np.zeros([n, 1])
        c2 = np.ones([n, 1])
        c3 = np.vstack((c1, c2))
        c4 = -1*c3
        c = np.vstack((c3, c4))
        c_mod = np.vstack((c, self.C_const))
        c_initial = np.vstack((c_mod, 0*e2n))
        'definition of initial x'
        x = np.zeros([len(c_initial), 1])
        'definition of initial basic_idx'
        basic_idx = np.arange(2*n)
        'definition of initial B and B_inv'
        B = A_initial[:, basic_idx]
        B_inv = np.linalg.inv(B)
        'initial lp setup'
        self.old_lp = LinearProg()
        self.old_lp.build_lp(A_initial, B, b_initial, c_initial, basic_idx, B_inv)
        self.old_lp.x = x

    def sequential_update(self, Phi):
        p = np.size(Phi,1)
        Phi_neg = -1*Phi
        "build auxiliary matrix A_aux, TODO: write a separate function"
        n = self.old_lp.A.shape[1]
        m = self.old_lp.A.shape[0]
        A_aux = np.zeros([2, n+2])
        A_aux[0, np.arange(0, p)] = Phi
        A_aux[0, np.arange(2*p, 3*p)] = Phi_neg
        A_aux[1, np.arange(0, p)] = Phi_neg
        A_aux[1, np.arange(2*p, 3*p)] = Phi
        A_aux[:, 4*p] = -1
        A_aux[0, n] = 1
        A_aux[1, n+1] = 1
        'determine if new constraint feasible'
        b1 = self.current_val
        b2 = -self.current_val
        x1 = b1 - A_aux[0, np.arange(0, n)].dot(self.old_lp.x)
        x2 = b2 - A_aux[1, np.arange(0, n)].dot(self.old_lp.x)
        if x1>-self.EPS and x2>-self.EPS:
            if self.verbosity > 1:
                print('new constraint feasible \n')
            self.new_lp = self.old_lp
            dual_simplex_iter = 0
        else:
            'TODO: write the following into a function update_lp'
            new_A = np.zeros([m+2, n+2])
            row_idx = np.arange(0, m)
            col_idx = np.arange(0, n)
            new_A[row_idx[:, None], col_idx] = self.old_lp.A
            new_A[[m, m+1], :] = A_aux
            new_basic_idx = np.hstack((self.old_lp.old_basic_idx, np.array([n, n+1])))
            new_B = new_A[:, new_basic_idx]
            new_B_inv = np.hstack((np.vstack((self.old_lp.B_inv, -A_aux[:, self.old_lp.old_basic_idx])), np.vstack((np.zeros([m, 2]), np.eye(2)))))
            new_b = np.vstack((self.old_lp.b, np.array([[b1], [b2]])))
            new_c = np.vstack((self.old_lp.c, np.array([[0], [0]])))
            new_x = np.vstack((self.old_lp.x, np.reshape(np.array([[x1], [x2]]), [2, -1])))
            self.new_lp = LinearProg()
            self.new_lp.build_lp(new_A, new_B, new_b, new_c, new_basic_idx, new_B_inv)
            self.new_lp.x = new_x
            if self.verbosity>1:
                print('running dual simplex')
            self.new_lp.run_dual_simplex()
            if self.new_lp.keep_flag:
                if self.verbosity>1:
                    print('new solution optimal, update lp')
                dual_simplex_iter = self.new_lp.iter_number
            else:
                self.new_lp = self.old_lp
                dual_simplex_iter = self.old_lp.MAX_ITERS
                if self.verbosity>1:
                    print('use original LP\n')
            self.old_lp = self.new_lp
        return dual_simplex_iter

    def build_phi_matrix(self):
        centers = self.centers
        embed_dim = centers.shape[0]
        center_num = centers.shape[1]
        Phi_mat = np.zeros((1, 1+center_num))
        for i in range(0, center_num):
            self.set_c(centers[:, i])
            Phi_mat[0, i+1] = self.get_phi_times_z()
        Phi_mat[0, 0] = 1
        return Phi_mat

    def get_weights(self):
        A = self.old_lp.A
        [m, n] = np.shape(A)
        x_1d = self.old_lp.x.ravel()
        x_intermediate = x_1d[np.arange(0, n-m)]
        x_eps = x_intermediate[-1]
        x_pure = x_intermediate[:-1]
        x_u = x_pure[np.arange(0, int((n-m-1)//2))]
        x_v = x_pure[int(n-m-1)//2:]
        x_free = x_u - x_v
        weights = np.concatenate((x_free[np.arange(0, int(n-m-1)//4)], [x_eps]))
        return weights

    def prune_center(self, sorted_weights, sorted_idx):
        DerivativeCutOff = self.DerivativeCutOff
        Nc = self.n-1
        derivatives = np.zeros(Nc-1)
        ratio = np.zeros(Nc-1)
        for i in range(0, Nc-1):
            if sorted_weights[i+1]<1e-15 and sorted_weights[i] < 1e-15:
                ratio[i] = 1
            elif sorted_weights[i+1]<1e-15 and sorted_weights[i] > 1e-15:
                ratio[i] = 100
            else:
                ratio[i] = sorted_weights[i]/sorted_weights[i+1]
            derivatives[i] = sorted_weights[i] - sorted_weights[i+1]
        findPos = False
        while not findPos:
            pos = np.argmax(ratio)
            if derivatives[pos] > DerivativeCutOff:
                center_idx = sorted_idx[:pos]
                findPos = True
            else:
                ratio[pos] = 0
        return center_idx



    def run(self):
        try:
            from .functions import GaussianRbf,ArcTan
        except:
            from functions import GaussianRbf,ArcTan
        #

        # if the user hasn't set any of the shape functions, use defaults.
        if not self.phi:    #test for existence
            self.phi = GaussianRbf()
        if not self.z:
            self.z = ArcTan()
        #

        if len(self.centers)==0:
            self.centers = self.generate_centers()
        #

        '''run preprocessing and sequential simplex'''
        # pre-processing
        pre_data = PreProcessing(self.y, self.t, self.smooth_time, self.embed_dim, self.predict_time)
        data, time = pre_data.clean()
        lag = pre_data.autocorrelation(data)
        input_data, output_data, index = pre_data.time_delayed_embed(data, lag, time)
        self.actual_t = index
        num_data_pts = len(index)
        # inital lp
        self.build_initial_lp()
        # sequential update
        for k in range(0, num_data_pts):
            if self.verbosity>0:
                print ('iteration %s'%(k))
            self.current_val = output_data[k]
            current_x = input_data[:, k]
            self.phi.x = current_x
            self.z.x = current_x
            Phi = self.build_phi_matrix()
            iter_k = self.sequential_update(Phi)
            if self.verbosity>1:
                print('dual simplex costs %s iteration\n' % iter_k)
            self.niter = np.concatenate((self.niter, [iter_k]))
            weights = self.get_weights()
            current_eps = weights[-1]

            if self.verbosity>1:
                print('current eps is %s' % current_eps)
            pure_weights = weights[:-1]
            self.eps_timeseries = np.concatenate((self.eps_timeseries, [current_eps]))
            sorted_weights = np.sort(np.abs(pure_weights[1:]))[::-1]
            sorted_idx = np.argsort(-np.abs(pure_weights[1:]))
            if all(kk < 1e-14 for kk in np.abs(pure_weights[1:])):
                if self.verbosity>1:
                    print('all weights are zeros, 0 rbf')
                current_nrbf = 0
                self.nrbf_timeseries = np.concatenate((self.nrbf_timeseries, [current_nrbf]))
            else:
                center_index = self.prune_center(sorted_weights, sorted_idx)
                self.nrbf_timeseries = np.concatenate((self.nrbf_timeseries, [len(center_index)]))
                if self.verbosity>1:
                    print('current iteration needs %s rbfs' % len(center_index))

    def visualize(self):
        '''
        Visualize the results; plot time series of
        epsilon, number of rbfs, number of iterations in the primal/dual solver.
        '''
        from matplotlib import pyplot

        # Would like these to be the same size as srbf.y;
        # fill appropriately with np.nan if needed.
        eps_thist = self.eps_timeseries # related to extract_parameter
        nrbf_thist = self.nrbf_timeseries #related to prune_center
        niter_thist = self.niter
        # Do an example plot of the results!
        fig,ax = pyplot.subplots(3,1, figsize=(12,6), sharex=True)

        t_actual = self.actual_t
        for i in range(3):
            ax[i].plot(self.t,self.y, c='k')
        #

        ax0r = ax[0].twinx()
        ax0r.plot(t_actual,eps_thist, c='r')

        ax1r = ax[1].twinx()
        ax1r.plot(t_actual,nrbf_thist, c='b')

        ax2r = ax[2].twinx()
        ax2r.bar(t_actual,niter_thist, width=t_actual[1]-t_actual[0], color='g')

        pyplot.show(block=False)

        return
    #
#


# -------- Test example --------------
if __name__=="__main__":
    import numpy as np
    from matplotlib import pyplot
    from preprocessing import *
    # from GaussianRbf import *
    # from ArcTanShapeFunction import *
    import pdb

    t = np.linspace(0, 8*np.pi, 801)
    y = np.sin(t) + 0.05*np.random.randn(len(t))


    # Instantiate the class
    srbf = SequentialRbf(200)
    srbf.y = y
    srbf.t = t
    srbf.C_const = 50.
    srbf.verbosity = 1
    # setting rbf and shape function
    # rbf_fun = GaussianRbf()
    # shape_fun = ArcTan()
    # srbf.set_functions(rbf_fun, shape_fun)
    # setting of embedding parameter
    # srbf.embed_dim = 3
    # srbf.smooth_time = 0
    # srbf.predict_time = 1
    # srbf.centers = srbf.generate_centers()


    # Run the full algorithm with a single call.
    srbf.run()
    # Would like these to be the same size as srbf.y;
    # fill appropriately with np.nan if needed.
    eps_thist = srbf.eps_timeseries # related to extract_parameter
    nrbf_thist = srbf.nrbf_timeseries #related to prune_center
    niter_thist = srbf.niter
    # Do an example plot of the results!
    fig,ax = pyplot.subplots(3,1, figsize=(12,6), sharex=True)

    t_actual = srbf.actual_t
    for i in range(3):
        ax[i].plot(t,y, c='k')
    #

    ax0r = ax[0].twinx()
    ax0r.plot(t_actual,eps_thist, c='r')

    ax1r = ax[1].twinx()
    ax1r.plot(t_actual,nrbf_thist, c='b')

    ax2r = ax[2].twinx()
    ax2r.bar(t_actual,niter_thist, width=t_actual[1]-t_actual[0], color='g')


    fig.tight_layout()
    pyplot.show(block=False)
    fig.savefig('SequentialRbf_example.png', dpi=120, bbox_inches='tight')
#
