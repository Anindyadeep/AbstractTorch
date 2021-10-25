try:
    import torch.nn as nn
    from Layers.layer_profiles import LAYER_PROFILE
    print("=====> (Activation Module) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f"{e} , Install all modules properly ...")


class _Activation(nn.Module):
    def __init__(self, activation, layer_profile):
        super(_Activation, self).__init__()
        self._layer_profile = layer_profile
        self._activation = activation
        self._activation_functions = {
            "relu"          :  nn.ReLU(),
            "relu6"         :  nn.ReLU6(),
            "selu"          :  nn.SELU(),
            "sigmoid"       :  nn.Sigmoid(),
            "log_sigmoid"   :  nn.LogSigmoid(),
            "tanh"          :  nn.Tanh(),
            "softmax"       :  nn.Softmax(),
            "softmax2d"     :  nn.Softmax2d(),
            "log_softmax"   : nn.LogSoftmax(),
        }
    
    def _getActivationFunc(self):
        layer = self._activation + "_layer_i"
        self._layer_profile[layer] += 1
        return self._activation_functions[self._activation]