from Layer import Layer
from ReLU import ReLU
from Sigmoid import Sigmoid
import numpy as np
import random


class Dense(Layer):
    def __init__(self, units, activation='relu', input_size=0, params=None):
        self.type = "dense"
        self.units = units
        self.activation = activation
        if isinstance(input_size, np.ndarray) or isinstance(input_size, tuple):
            self.input_size = input_size[0]
        else:
            self.input_size = input_size
        self.dednet = 0
        self.dedx = 0
        self.dedw = 0
        self.output = np.zeros(units)
        self.params = params
        if params is not None:
            self.weight = np.concatenate((np.array(params['kernel']),[np.array(params['bias'])]),axis=0).transpose()
        else:
            self.weight = None
        if self.weight is not None:
            self.parameters = self.weight.flatten().size
        else:
            self.parameters = 0

    def initialize_input(self, input):
        self.input = np.concatenate([input, [1]])
        self.input_size = self.input.shape[0]

    def initialize_weight(self):
        if self.weight is None:
            self.weight = np.ndarray((self.units, self.input_size))
            for i in range(self.units):
                for j in range(self.input_size):
                    self.weight[i,j] = random.randrange(-10, 10)

    def initialize_output(self):
        self.output = np.ndarray(self.units)

    def forward_prop(self):
        for i in range(self.units):
            self.output[i] = 0
            for j in range(self.input.shape[0]):
                self.output[i] += self.weight[i,j]*self.input[j]

            self.activation_function = ReLU()
            if self.activation == "relu":
                self.activation_function = ReLU()
            elif self.activation == "sigmoid":
                self.activation_function = Sigmoid()
            self.output[i] = self.activation_function.calculate(self.output[i])

    def run(self, input):
        self.initialize_input(input)
        self.initialize_weight()
        self.initialize_output()
        self.forward_prop()
        self.parameters = self.weight.flatten().size

    def back_prop(self,e):
        if(not isinstance(e, np.ndarray)): e = np.ones(self.units)*e
        else: self.dedo = np.copy(e)

        self.dednet = np.copy(self.output)
        for i in range(self.units):
            self.dednet[i] = self.dedo[i]*self.activation_function.calculate_derivative(self.dednet[i])

        self.dedw = np.ndarray(self.input_size, self.units)
        for i in range(self.input_size):
            for j in range(self.units):
                self.dedw[i,j] = self.input[i]*self.dednet[j]

        self.dedx = np.ndarray(self.input_size-1)
        for i in range(self.input_size-1):
            self.dedx[i] = 0
            for j in range(self.units):
                self.dedx[i] += self.dednet[j]*self.weight[i+1,j]

            def update_weight(self, rate):
                for i in range(self.units):
                    for j in range(self.input_size):
                        self.weight[i,j] += rate*self.dedw[i,j]

    def print_summary(self):
        print(f"Layer Type : Dense")
        print(f"Output size : {self.output.shape}")
        print(f"Parameters : {self.parameters}")
        print(f"Weight:\n{self.weight}")