import numpy as np
import pdb

class LinearProg:
    def __init__(self):
        self._A = None
        self._B = None
        self._i = None
        self._j = None
        self.old_basic_idx = None
        self._new_basic_idx = None
        self.non_basic_idx = None
        self._B_inv = None
        self._c = None
        self._c_bar = None
        self._b = None
        self.x = None
        self._x_B = None
        self._optimal_flag = None
        self._EPS = 1e-10
        self.MAX_ITERS = 100
        self._l = None
        self._i = None
        self.iter_number = None
        self.keep_flag = True

        self.verbosity = 0

    def build_lp(self, A, B, b, c, old_basic_idx, B_inv):
        self._A = A
        self._B = B
        self._b = b
        self._c = c
        self.old_basic_idx = old_basic_idx
        self._B_inv = B_inv
        self._B_inv = np.linalg.inv(B)

    def _compute_c_bar(self):
        num_vars = self.A.shape[1]
        self.non_basic_idx = np.setdiff1d(np.arange(0, num_vars), self.old_basic_idx)
        c_B = self._c[self.old_basic_idx, 0]
        self._c_bar = self._c[self.non_basic_idx].T-c_B.T.dot(self._B_inv).dot(self._A[:, self.non_basic_idx])
        # blerp = [i > -self._EPS for i in self._c_bar[0]] # Note self._c_bar is a row vector with two dimensions.
        blerp = self._c_bar > -self._EPS
        self._c_bar[np.where(np.abs(self._c_bar) < self._EPS)] = 0
        assert np.all(blerp), 'c_bar should be non-negative i.e. solution should be optimal\n'

    def _compute_l(self):
        self._x_B = self._B_inv.dot(self._b)
        x_B = self.x[self.old_basic_idx]
        if all(i > -self._EPS for i in self._x_B):
            self._optimal_flag = True
        else:
            negative_idx = np.where(self._x_B < -self._EPS)[0]
            self._l = negative_idx[0]

    def _compute_j(self):
        B_inv_l = self._B_inv[self._l, :]
        v = B_inv_l.dot(self._A[:, self.non_basic_idx])
        if np.all(v > -self._EPS):   # NOTE THAT THIS ISN'T A SCALAR. NP.ALL WILL WORK REGARDLESS.
            assert False, 'optimal dual cost is infinity\n'
        else:
            negative_idx = np.where(v < -self._EPS)[0]
            v_neg = v[negative_idx]
            c_bar_neg = self._c_bar[0, negative_idx]
            ratio = np.divide(c_bar_neg, np.abs(v_neg))
            ratio[np.where(ratio < self._EPS)] = 0
            j_tmp = np.argmin(ratio)
            self._j = self.non_basic_idx[negative_idx[j_tmp]]

    def _update_basic_idx(self):
        self._new_basic_idx = self.old_basic_idx
        self._new_basic_idx[self._l] = self._j
        self.old_basic_idx = self._new_basic_idx

    def _compute_B_inv(self):
        m = self._B_inv.shape[0]
        u = self._B_inv.dot(self._A[:, self._j])
        Q = np.eye(m)
        Q_l = -np.divide(u, u[self._l])
        Q_l[self._l] = np.divide(-Q_l[self._l], u[self._l])
        Q[:, self._l] = Q_l
        self._B_inv = Q.dot(self._B_inv)

    def run_dual_simplex(self):
        count = 0
        while count < self.MAX_ITERS or self._optimal_flag:
            self._compute_c_bar()
            self._compute_l()
            if self._optimal_flag:
                break
            else:
                self._compute_j()
                self._update_basic_idx()
                self._compute_B_inv()
                self.x = np.zeros((len(self._c), 1))
                self.x[self.old_basic_idx] = self._B_inv.dot(self._b)
            count += 1
        if count >= self.MAX_ITERS-1:
            self.keep_flag = False
            if self.verbosity>0:
                print("Maximum number of iterations reached!\n")
        self.iter_number = count

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def B_inv(self):
        return self._B_inv

    @B_inv.setter
    def B_inv(self, value):
        self._B_inv = value
