#!/usr/bin/env python3
"""
A debugger that records the result of the simulation
"""

class Debugger():
    def __init__(self):
        self.time = []

        self.com_pin = []
        self.com_pb = []
        self.com_ref = []

        self.com_pin_vel = []
        self.com_pb_vel = []
        self.com_ref_vel = []
        
        self.com_pin_acc = []
        self.com_pb_acc = []
        self.com_ref_acc = []

        self.ZMP_plan = []
        self.ZMP_est = []
        self.DCM_est = []

        self.lf_pos = []
        self.lf_vel = []
        self.lf_acc = []

        self.rf_pos = []
        self.rf_vel = []
        self.rf_acc = []
        
        self.lf_normal = []
        self.rf_normal = []

        self.angularMoment = []
        self.refMoment = []
    
    def record(self, robot, interpolator):
        pass

    def plot(self):
        pass