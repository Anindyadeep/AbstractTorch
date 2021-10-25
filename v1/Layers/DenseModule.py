from torch.nn.modules import activation

try:
    import sys 
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent 
    sys.path.append(str(BASE_DIR))
    import torch.nn as nn
    from Layers.layer_profiles import LAYER_PROFILE
    from Layers.ActivationModule import _Activation
    print("=====> (Dense Module) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f" ERROR: {e} Install all modules properly ...")


class _Flatten(nn.Module):
    def __init__(self, layer_profile, arch_profile, x):
        super(_Flatten, self).__init__()
        self._layer_profile = layer_profile
        self._arch_profile = arch_profile
        self._x = x
    
    def _perform_flatten(self):
        self._layer_profile["isflatten"] = True
        flatten = nn.Flatten()
        self._x_flatten = flatten(self._x)
        self._arch_profile["flatten"] = flatten
        return [
            self._layer_profile,
            self._arch_profile,
            self._x_flatten
        ]


class _Dense(nn.Module):
    def __init__(self, layer_profile, arch_profile, params_profile, x, neurons, activation=None, dropout=None):
        super(_Dense, self).__init__()
        self._layer_profile = layer_profile
        self._arch_profile = arch_profile
        self._params_profile = params_profile
        self._x = x
        self._neurons = neurons
        self._in_features = 0
        self._activation = activation
        self._dropout = dropout
    
    def _perform_dense(self):
        self._layer_profile["linear_layer_i"] += 1
        if self._layer_profile["linear_layer_i"] == 0:
            if self._layer_profile["isflatten"]:
                self._in_features = self.x.shape[1]
            else:
                print("Not flattened ... flattening the layer")
                local_flatten = _Flatten(self._layer_profile, self._arch_profile, self._x)
                self._layer_profile, self._arch_profile, self._x = local_flatten._perform_flatten()
                self._in_features = self._x.shape[1]
        else:
            previous_layer = "linear_layer_" + str(int(self._layer_profile["linear_layer_i"]-1))
            self._in_features = self._arch_profile[previous_layer].out_features
        
        linear = nn.Linear(
                in_features = self._in_features,
                out_features = self._neurons
        )
        
        self._x_dense = linear(self._x)
        current_layer = "linear_layer_" + str(int(self._layer_profile["linear_layer_i"]))
        self._arch_profile[current_layer] = linear

        if self._activation:
            activation_layer = _Activation(self._activation, self._layer_profile)._getActivationFunc()
            current_activation_layer_str = self._activation + "_within_dense_" + str(int(self._layer_profile["linear_layer_i"]))
            self._arch_profile[current_activation_layer_str] = activation_layer
        
        if self._dropout:
            self._layer_profile["dropout1d_layer_i"] += 1
            dropout1d = nn.Dropout(p = self._dropout, inplace = True)
            current_dropout_layer_str = "dropout1d_within_linear_" + str(int(self._layer_profile["linear_layer_i"]))
            self._arch_profile[current_dropout_layer_str] = dropout1d
        
        # calculate the number of parameters 

        if self._layer_profile["linear_layer_i"] > 0 and self._layer_profile["dropout1d_layer_i"] >= 0:
            if self._layer_profile["dropout1d_layer_i"] < self._layer_profile["linear_layer_i"]:
                p = self._dropout
            else: 
                p = 0
        else: 
            p = 0
        
        # ( ========= BUG ========== )

        batch_size = self._x.shape[0]
        previous_num_params = self._in_features
        current_num_params =  int(self._neurons * p)
        params_linear = (batch_size * previous_num_params) + current_num_params

        if self._dropout:
            self._params_profile["linear_layer_" + str(int(self._layer_profile["linear_layer_i"])) + "_dropout1d"] = params_linear
        else: 
            self._params_profile["linear_layer_" + str(int(self._layer_profile["linear_layer_i"]))] = params_linear

        return [
            self._layer_profile,
            self._arch_profile,
            self._params_profile,
            self._x_dense
        ]


if __name__ == '__main__':
    import pandas as pd 
    from tabulate import tabulate
    import torch 
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.rand(256, 256, 58, 58)
    y = torch.rand(256)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    x, y = next(iter(loader))

    main_layer_profile = LAYER_PROFILE
    main_arch_profile = {}
    main_params_profile = {}

    test_dense1 = _Dense(
        layer_profile = main_layer_profile,
        arch_profile = main_arch_profile,
        params_profile = main_params_profile,
        x = x,
        neurons = 1024,
        activation = "relu"
    )

    test_lpd1, test_apd1, test_ppd1, test_x_dense1 = test_dense1._perform_dense()

    test_dense2 = _Dense(
        layer_profile = main_layer_profile,
        arch_profile = main_arch_profile,
        params_profile = main_params_profile,
        x = test_x_dense1,
        neurons = 256,
        activation = "relu",
        dropout = 0.2
    )

    test_lpd2, test_apd2, test_ppd2, test_x_dense2 = test_dense2._perform_dense()

    def table(dictionary):
        print_dict = {}
        for key in dictionary.keys():
            print_dict[key] = str(dictionary[key])
        
        df = pd.DataFrame(
        {
            "LAYERS" : list(print_dict.keys()),
            "LAYER_DESC" : list(print_dict.values())
        })

        print("\n")
        return tabulate(df, headers=df.columns)

    print(table(test_apd2))
    print(table(test_ppd2), "\n")

    dense_module = nn.ModuleDict(test_apd2)
    print(dense_module, "\n")

    import torch.optim as optim 
    optimizer = optim.Adam(dense_module.parameters(), lr = 0.01)
    print(optimizer)
