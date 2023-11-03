from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def calculate(self, input):
        pass

    def calculate_derivative(self, input):
        pass