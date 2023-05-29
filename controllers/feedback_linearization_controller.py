import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.kp = 25
        self.kd = 60
        self.kp = np.array([[self.kp, 0], [0, self.kp]])
        self.kd = np.array([[self.kd, 0], [0, self.kd]])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        M = self.model.M(x)
        C = self.model.C(x)
        q = x[:2]
        q_dot = x[2:]

        v = q_r_ddot + self.kd@(q_r_dot - q_dot) + self.kp@(q_r - q)
        tau= M@v[:,np.newaxis] + C@q_dot[:,np.newaxis]

        return tau
