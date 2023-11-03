import json
from json import JSONEncoder
import numpy as np
from Sequential import Sequential
from Dense import Dense
from LSTM import LSTM

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Loader:
    def save(self, path, model):
        to_save = []
        for layer in model.layers:
            if(layer.type == "lstm"):
                layer_dictionary = {
                    "type": "lstm",
                    "units": layer.units,
                    "input_shape": layer.input_shape,
                    "params": layer.params
                }
            elif layer.type == "dense":
                layer_dictionary = {
                    "type": "dense",
                    "units": layer.units,
                    "input_size": layer.input_size,
                    "activation": layer.activation,
                    "params": layer.params
                }
            to_save.append(layer_dictionary)


        outfile = open(path, "w")
        json.dump(to_save, outfile, indent=4, cls=NumpyArrayEncoder)
        outfile.close()

    def load(self, path):
        f = open(path)

        data = json.load(f)

        model = Sequential([])
        for d in data:
            if d["type"] == "dense":
                layer = Dense(d["units"], activation=d["activation"], input_size=d["input_size"], params=d["params"])
            elif d["type"] == "lstm":
                layer = LSTM(d["units"], d["input_shape"], d["params"])
            model.add(layer)
        return model