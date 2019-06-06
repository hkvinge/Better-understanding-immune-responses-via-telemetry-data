import numpy as np
from RBF_function import ShapeFunction

class ArcTan(ShapeFunction):
    def __init__(self, Lambda=np.zeros((3, 1)), x=np.zeros((3, 1)), c=np.zeros((3, 1))):
        ShapeFunction.__init__(self, Lambda, x, c)

    def get_val(self):
        return np.arctan(self.calculate_rho())/np.pi+0.5

    def z_dash(self):
        return 1/(np.pi*(1+self.calculate_rho()*self.calculate_rho()))