class Sequential():
    def __init__(self, layers, batch_size=1):
        self.layers = layers
        self.batch_size = batch_size

    def fit(self, X_train, y_train, epochs = 10, batch_size=1):
        target = 0
        for i in range(len(y_train)):
            out = 1 if (y_train[i] == "panda") else 0
            e = - (target - out)
            target=self.predict(X_train[i])

            for j in range(len(self.layers)-1,0):
                e = self.layers[j].back_prop(e)


    def test(self, input, layer):
        layer.run(input)
        return layer.output

    def predict(self, input):
        for layer in self.layers:
            input = self.test(input, layer)
        return input

    def evaluate(self, X_test, y_test):
        total = y_test.size[0]
        correct = 0
        for i in range(y_test.shape[0]):
            if self.predict(X_test[i]) == y_test[i]:
                correct+=1

    def add(self, layer):
        self.layers.append(layer)

    def summarize(self):
        for i in range(len(self.layers)):
            print(f"Layer {i+1}")
            self.layers[i].print_summary()
            print()
