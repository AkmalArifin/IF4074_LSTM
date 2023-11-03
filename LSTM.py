from Layer import Layer
from Cell import Cell
import numpy as np

class LSTM(Layer):
    def __init__(self, units, input_shape, params):
        self.type = "lstm"
        self.units = units
        self.input_shape = input_shape
        self.params = params
        self.weights = self.initialize_weight(params)
        self.cell = Cell(units, self.weights)
        self.parameters = np.array(self.weights).flatten().size
        self.output = self.cell.output

    def initialize_weight(self, params):
        W_i = params["W_i"]
        W_f = params["W_f"]
        W_c = params["W_c"]
        W_o = params["W_o"]
        U_i = params["U_i"]
        U_f = params["U_f"]
        U_c = params["U_c"]
        U_o = params["U_o"]
        b_i = params["b_i"]
        b_f = params["b_f"]
        b_c = params["b_c"]
        b_o = params["b_o"]

        weights_i = []
        weights_f = []
        weights_c = []
        weights_o = []

        # get i
        for i in range(self.units):
            temp = []
            for w in W_i:
                temp.append(w[i])
            for u in U_i:
                temp.append(u[i])
            temp.append(b_i[i])
            weights_i.append(temp)

        # get f
        for i in range(self.units):
            temp = []
            for w in W_f:
                temp.append(w[i])
            for u in U_f:
                temp.append(u[i])
            temp.append(b_f[i])
            weights_f.append(temp)

        # get c
        for i in range(self.units):
            temp = []
            for w in W_c:
                temp.append(w[i])
            for u in U_c:
                temp.append(u[i])
            temp.append(b_c[i])
            weights_c.append(temp)

        # get o
        for i in range(self.units):
            temp = []
            for w in W_o:
                temp.append(w[i])
            for u in U_o:
                temp.append(u[i])
            temp.append(b_o[i])
            weights_o.append(temp)

        return [weights_f, weights_i, weights_c, weights_o]

    def run(self, input):
        for i in range(self.input_shape[0]):
            self.cell.run(input[i])
        self.output = self.cell.output

    def print_summary(self):
        print(f"Layer Type : LSTM")
        print(f"Output size : {self.output.shape}")
        print(f"Parameters : {self.parameters}")
        print(f"Weight:\n{np.array(self.weights)}")