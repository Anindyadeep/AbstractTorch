try:
    import sys 
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent 
    sys.path.append(str(BASE_DIR))
    import torch.nn as nn
    from Layers.layer_profiles import LAYER_PROFILE
    from Layers.ActivationModule import _Activation
    print("=====> (Conv Module) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f" ERROR: {e} Install all modules properly ...")

"""
TODO
----

1. Add different types of Conv layer based on the dimension

"""

class _Conv2d(nn.Module):
    def __init__(self, layer_profile, arch_profile, params_profile, x, num_filters, kernel_size, padding = 1, stride = 1, dilation = (1,1), activation = None, dropout = None):
        super(_Conv2d, self).__init__()
        self._layer_profile = layer_profile
        self._arch_profile = arch_profile
        self._x = x 
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride 
        self._dilation = dilation
        self._in_channels = -1
        self._activation = activation
        self._dropout = dropout
        self._params_profile = params_profile
    
    def _perform_conv2d(self):
        self._layer_profile["conv2d_layer_i"] += 1
        if self._layer_profile["conv2d_layer_i"] == 0:
            self._in_channels = self._x.shape[1]
        else:
            previous_channel_str = "conv2d_layer_" + str(int(self._layer_profile["conv2d_layer_i"]-1))
            self._in_channels = self._arch_profile[previous_channel_str].out_channels
        
        conv2d = nn.Conv2d(
            in_channels = self._in_channels,
            out_channels = self._num_filters,
            kernel_size = self._kernel_size,
            stride = self._stride,
            padding = self._padding,
            dilation = self._dilation
        )

        self._x_conv2d = conv2d(self._x)
        current_layer_str = "conv2d_layer_" + str(int(self._layer_profile["conv2d_layer_i"]))
        self._arch_profile[current_layer_str] = conv2d

        if self._activation:
            activation_layer = _Activation(self._activation, self._layer_profile)._getActivationFunc()
            current_activation_layer_str = self._activation + "_within_conv2d_" + str(int(self._layer_profile["conv2d_layer_i"]))
            self._arch_profile[current_activation_layer_str] = activation_layer
        
        if self._dropout:
            self._layer_profile["dropout2d_layer_i"] += 1
            dropout2d = nn.Dropout2d(p = self._dropout)
            current_dropout_layer_str = "dropout2d_within_conv2d_" + str(int(self._layer_profile["conv2d_layer_i"])) 
            self._arch_profile[current_dropout_layer_str] = dropout2d

        
        # calculating the parameters of the conv
        
        if type(self._kernel_size) == int:
            width = self._kernel_size
            hight = self._kernel_size
        
        else: 
            hight = self._kernel_size[0]
            width = self._kernel_size[1]
        
        prev_filter_num = self._in_channels
        curr_filter_num = self._num_filters

        if self._dropout:
            p = self._dropout
        else: 
            p = 1
        
        batch_size = self._x.shape[0]
        params_conv2d = (((width * hight) * int(prev_filter_num * 1-p)) + 1) * curr_filter_num

        # ( ========= BUG ========== )

        if self._dropout:
            self._params_profile["conv2d_layer_" + str(int(self._layer_profile["conv2d_layer_i"])) + "_dropout2d"] = int(batch_size * params_conv2d)
        else: 
            self._params_profile["conv2d_layer_" + str(int(self._layer_profile["conv2d_layer_i"]))] = int(batch_size * params_conv2d)

        return [
            self._layer_profile,
            self._arch_profile,
            self._params_profile,
            self._x_conv2d,
        ]



# CONV TESTING
if __name__ == '__main__':
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent 
    sys.path.append(str(BASE_DIR))
    import torch 
    from torch.utils.data import DataLoader, TensorDataset
    from tabulate import tabulate
    import pandas as pd 
    import json
    from Layers.layer_profiles import LAYER_PROFILE

    X = torch.rand(256, 3, 64, 64)
    y = torch.rand(256)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    x, y = next(iter(loader))

    main_layer_profile = LAYER_PROFILE
    main_arch_profile = {}
    main_params_profile = {}

    
    test_conv1 = _Conv2d(
        layer_profile = main_layer_profile,
        arch_profile = main_arch_profile,
        params_profile = main_params_profile,
        num_filters = 128,
        x = x,
        kernel_size = (5,5),
        activation = "relu",
        #dropout = 0.3
    )
    test_lp1, test_ap1, test_pp1, test_x_conv2d1 = test_conv1._perform_conv2d()

    test_conv2 = _Conv2d(
        layer_profile = test_lp1,
        arch_profile = test_ap1,
        params_profile = test_pp1,
        num_filters = 256,
        x = test_x_conv2d1,
        kernel_size = (7,7),
        activation = "relu"
    )
    test_lp2, test_ap2, test_pp2, test_x_conv2d2 = test_conv2._perform_conv2d()

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


    import json 
    print(table(test_ap2))
    print(table(test_pp2))
    print("\n")
    conv_module = nn.ModuleDict(test_ap2)
    print(conv_module)