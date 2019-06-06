import numpy as np
# from RBF_function import RbfFunction

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

class GaussianRbf(RbfFunction):
    def __init__(self, A = np.eye(3), x=np.zeros((3, 1)), c=np.zeros((3, 1)), alpha=0.5):
        RbfFunction.__init__(self,A,x,c)
        self.alpha = alpha

    def get_val(self):
        return np.exp(-self.calculate_mu()*self.calculate_mu() / (self.alpha * self.alpha))

    def phi_dash(self):
        return (-2*self.calculate_mu()/(self.alpha*self.alpha))*np.exp(-(self.calculate_mu()*self.calculate_mu())/(self.alpha**2))

    def dphi_dalpha(self):
        return (2*self.calculate_mu()*self.calculate_mu() / (self.alpha ** 3)) * np.exp(-(self.calculate_mu()*self.calculate_mu())/(self.alpha**2));

class ArcTan(ShapeFunction):
    def __init__(self, Lambda=np.zeros((3, 1)), x=np.zeros((3, 1)), c=np.zeros((3, 1))):
        ShapeFunction.__init__(self, Lambda, x, c)

    def get_val(self):
        return np.arctan(self.calculate_rho())/np.pi+0.5

    def z_dash(self):
        return 1/(np.pi*(1+self.calculate_rho()*self.calculate_rho()))
