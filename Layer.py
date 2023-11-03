from abc import ABC, abstractmethod

class Layer(ABC):
    def set_activation(self, activation):
        self.activation = activation
        self.parameters = 0

    @abstractmethod
    def run(self, input):
        pass