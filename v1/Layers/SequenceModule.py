try:
    import sys 
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent 
    sys.path.append(str(BASE_DIR))
    import torch.nn as nn
    from Layers.layer_profiles import LAYER_PROFILE
    from Layers.ActivationModule import _Activation
    print("=====> (Sequence Module) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f" ERROR: {e} Install all modules properly ...")


class _RNN(nn.Module):
    def __init__(self, layer_profile, arch_profile, params_profile, x, input_size, hidden_size, num_layers, activation, bias, batch_first, dropout, bi_directional):
        super(_RNN, self).__init__()
        self._layer_profile = layer_profile
        self._arch_profile = arch_profile
        self._params_profile = params_profile
        self._x = x
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._non_linearity = activation,
        self._bias = bias,
        self._batch_first = batch_first,
        self._dropout = dropout
        self._bi_directional = bi_directional
    
    def _perform_rnn(self):
        pass 