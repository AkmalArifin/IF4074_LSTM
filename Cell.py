import math
import numpy as np
from Function import sigmoid

# ukuran weights : 4 kali jumlah fitur + jumlah unit
# ukuran biases : 4

class Cell():
    def __init__(self, units, weights, cell_state=0, output=0):
        if cell_state==0:
            cell_state = np.array([0. for i in range(units)])
        if output==0:
            output = np.array([0. for i in range(units)])

        self.weights = weights
        self.cell_state = cell_state    # jumlah unit
        self.output = output            # jumlah unit

    def forget_gate(self, input, weight):
        w_input = np.empty(len(weight), dtype=float)
        for i in range(len(weight)):
            w_input[i] = np.dot(input, weight[i])

        ft = np.empty(len(w_input), dtype=float)
        for i in range(len(w_input)):
            ft[i] = sigmoid(w_input[i])
        return ft

    def input_gate(self, input, weight):
        w_input = np.empty(len(weight), dtype=float)
        for i in range(len(weight)):
            w_input[i] = np.dot(input, weight[i])

        it = np.empty(len(w_input), dtype=float)
        for i in range(len(w_input)):
            it[i] = sigmoid(w_input[i])
        return it

    def output_gate(self, input, weight):
        w_input = np.empty(len(weight), dtype=float)
        for i in range(len(weight)):
            w_input[i] = np.dot(input, weight[i])

        ot = np.empty(len(w_input), dtype=float)
        for i in range(len(w_input)):
            ot[i] = sigmoid(w_input[i])
        return ot

    def candidate(self, input, weight):
        w_input = np.empty(len(weight), dtype=float)
        for i in range(len(weight)):
            w_input[i] = np.dot(input, weight[i])

        ct = np.empty(len(w_input), dtype=float)
        for i in range(len(w_input)):
            ct[i] = math.tanh(w_input[i])
        return ct

    def run(self, input):
        input = np.append(input, self.output, axis=0)
        input = np.append(input, 1) # add bias
        ft = self.forget_gate(input, self.weights[0])
        self.cell_state *= ft

        it = self.input_gate(input, self.weights[1])
        ct = self.candidate(input, self.weights[2])
        new_cell_state = it * ct
        self.cell_state += new_cell_state

        ot = self.output_gate(input, self.weights[3])
        ct = np.empty(len(self.cell_state), dtype=float)
        for i in range(len(self.cell_state)):
            ct[i] = math.tanh(self.cell_state[i])
        self.output = ot * ct
