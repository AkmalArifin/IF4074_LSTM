from ActivationFunction import ActivationFunction
import math

class Sigmoid(ActivationFunction):
    def calculate(self, input):
        return 1/(1+math.exp(-input))

    def calculate_derivative(self, input):
        s = self.calculate(input)
        return s*(1-s)