import numpy as np


class RbfFunction:
    def __init__(self, A = np.eye(3), x=np.zeros((3, 1)), c=np.zeros((3, 1))):
        self._MU_EPS = 1e-15
        self.A = A
        self.x = x
        self.c = c

    'over write in subclass'
    def phi_dash(self):
        return 0
    'over write in subclass'
    def dphi_dalpha(self):
        return 0

    def get_val(self):
        return 0

    def calculate_mu(self):
        mu = self._x_minus_c().dot(self.A).dot(self._x_minus_c())
        return mu

    def dmu_dc(self):
        return -1*self.A.dot(self.A.T).dot(self._x_minus_c())/self.calculate_mu()

    def dmu_dA(self):
        v = np.reshape(self._x_minus_c(), [-1, 1])
        return self.A.dot(v).dot(v.T)/self.calculate_mu()

    def _x_minus_c(self):
        return self.x-self.c


class ShapeFunction:
    def __init__(self, Lambda, x, c):
        self.Lambda = Lambda
        self.x = x
        self.c = c

    'over write in subclass'
    def z_dash(self):
        return 0

    'over write in subclass'
    def get_val(self):
        return 0

    def drho_dc(self):
        return -1*self.Lambda

    def drho_dlambda(self):
        return self.x - self.c

    def calculate_rho(self):
        return self.Lambda.T.dot(self.x-self.c)




