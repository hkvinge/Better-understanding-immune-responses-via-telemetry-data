import numpy as np
from RBF_function import RbfFunction


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
