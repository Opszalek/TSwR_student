import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel as ManipulatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        model1=ManipulatorModel(Tp)
        model2=ManipulatorModel(Tp)
        model3=ManipulatorModel(Tp)
        model1.m3=0.1
        model1.r3=0.05
        model2.m3=0.01
        model2.r3=0.01
        model3.m3=1.0
        self.models = [model1, model2, model3]
        self.i = 0
        self.prev_u= np.zeros((2, 1))
        self.prev_x = np.zeros(4)
        self.kp = 50.0
        self.kd = 25.0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        xest=[]
        for model in self.models:
            xest.append(model.x_dot(self.prev_x, self.prev_u)-(1/self.Tp)*self.prev_x.reshape(4,1))
        model_error=[]
        for x_model in xest:
            model_error.append(np.linalg.norm(x_model-x))
        self.i=np.argmin(model_error)



    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + self.kd * e_dot + self.kp * e # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u=u
        self.prev_x=x
        return u
