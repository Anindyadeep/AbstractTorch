try:
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent 
    sys.path.append(str(BASE_DIR))
    from Layers.ConvModule import _Conv2d
    from Layers.DenseModule import _Dense
    from Layers.layer_profiles import LAYER_PROFILE
    import torch.nn as nn
    import torch
    import torch.optim as optim 
    from tabulate import tabulate
    import pandas as pd 
    print("=====> (Model Arch) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f"ERROR: {e}")
    
class Model(nn.Module):
    def __init__(self, model_profile):
        super(Model, self).__init__()
        self.model_profile = model_profile
        self.model = nn.ModuleDict(self.model_profile)
    
    def forward(self, x):
        for key in self.model.keys():
            x = self.model[key](x)
        return x


class TorchModel(nn.Module):
    def __init__(self, train_loader, valid_loader):
        super(TorchModel, self).__init__()
        self.layer_profile = LAYER_PROFILE
        self.params_profile = {}
        self.arch_profile = {}
        self.train_loader = train_loader 
        self.valid_loader = valid_loader
        self._x , self._y = next(iter(train_loader))
    
    def _table(self, dictionary):
        print_dict = {}
        for key in dictionary.keys():
            print_dict[key] = str(dictionary[key])
        
        df = pd.DataFrame(
        {
            "LAYERS" : list(print_dict.keys()),
            "SUMMARY" : list(print_dict.values())
        })

        print("\n")
        return tabulate(df, headers=df.columns) 

    def addConv2d(self, num_filters, kernel_size, padding = 1, stride = 1, dilation = (1,1), activation = None, dropout = None):
        """
        params:
        ------
        num_filters    : (int) The number of the filters required for the convolution
        kernel_size    : (int) or (tuple) The hight, width of the kernel required for convolution
        padding        : (int) padding if required , default value 1
        stride         : (int) the strides in convolution, default value 1
        dilation       : (int) or (tuple) default (1,1)
        activation     : (string) default (None) options : [relu, sigmoid, softmax, log_softmax, log_sigmoid, relu6, tanh]
        dropout        : (float)  default (None), provides dropout2d for the CNN layer as a whole new layer
        """
        conv2d = _Conv2d(
            layer_profile = self.layer_profile,
            arch_profile = self.arch_profile,
            params_profile = self.params_profile,
            num_filters = num_filters,
            x = self._x,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            activation = activation,
            dropout = dropout
        )

        self.layer_profile, self.arch_profile, self.params_profile, self._x = conv2d._perform_conv2d()
    
    def addDense(self, neurons, activation = None, dropout = 0):
        """
        params:
        ------
        neurons          : (int) The number of the hidden layer neurons in the network
        activation     : (string) default (None) options : [relu, sigmoid, softmax, log_softmax, log_sigmoid, relu6, tanh]
        dropout        : (float)  default (None), provides dropout1d for the ANN layer as a whole new layer
        """
        dense = _Dense(
            layer_profile = self.layer_profile,
            arch_profile = self.arch_profile,
            params_profile = self.params_profile,
            x = self._x,
            neurons = neurons,
            activation = activation,
            dropout = dropout)

        self.layer_profile, self.arch_profile, self.params_profile, self._x = dense._perform_dense()
    
    def addMaxPool2d(self, params):
        pass

    def addDropout2d(self, params):
        pass

    def addDropout(self, params):
        pass

    def addRNN(self, params):
        pass 

    def addLSTM(self, params):
        pass
    
    def addGRU(self, params):
        pass

    def addTransformer(self, params):
        pass

    def arch_summary(self):
        return self._table(self.arch_profile)

    def param_summary(self):
        return self._table(self.params_profile)
    
    def get_model(self):
        return Model(self.arch_profile)