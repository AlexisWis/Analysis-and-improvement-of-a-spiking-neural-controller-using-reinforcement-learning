import numpy as np
from typing import Tuple

class Physics():
    
    def __init__(self, g = 9.81, m_c = 1, m_p = 0.1, l = 0.5, dt = 0.02) -> None:
        self.__dict__.update(vars())
        self.x, self.v, self.theta, self.w = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.dw = 0


    def dw_step(self, cart_force: float) -> float:
        numerator = self.g * np.sin(self.theta) + np.cos(self.theta) * (-cart_force - self.m_p * self.l * self.w**2 * np.sin(self.theta))/(self.m_c+self.m_p)
        denominator = self.l * (4/3 - (self.m_p*np.cos(self.theta)**2)/(self.m_c+self.m_p))

        self.dw = numerator/denominator
        self.w += self.dt * self.dw
        self.theta += self.dt * self.w

        return self.theta
    
    def a_step(self, force: float) -> float:
        numerator = force + self.m_p * self.l * (self.w**2 * np.sin(self.theta) - self.dw * np.cos(self.theta))
        denominator = self.m_c + self.m_p

        self.a = numerator/denominator
        self.v += self.dt * self.a
        self.x += self.dt * self.v

        return self.x

    def update(self, force: float) -> Tuple[float, float]:
        return (self.dw_step(force), self.a_step(force))
    
    #get state of the system that agent can see
    def get_state(self) -> Tuple[float,float,float,float]:
        return (self.x, self.theta, self.v, self.w)
    
    def reset(self) -> None:
        self.x, self.v, self.theta, self.w = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.dw = 0
