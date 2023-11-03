from ActivationFunction import ActivationFunction

class ReLU(ActivationFunction):
    def calculate(self, input):
        return input if input > 0 else 0

    def calculate_derivative(self, input):
        return 0 if input<=0 else 1