import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate
import pandas as pd 

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

X = torch.rand(256, 3, 64, 64)
y = torch.rand(256)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

torch_model = TorchModel(loader, loader)
torch_model.addConv2d(256, (5,5), activation="relu")
torch_model.addConv2d(128, (7,7), activation = "relu")
torch_model.addConv2d(128, (3,3), dropout=0.1)
torch_model.addConv2d(64, (3,3), dropout = 0.1)

torch_model.addDense(1028, activation="relu", dropout = 0.1)
torch_model.addDense(256, activation="relu"),
torch_model.addDense(256, activation="relu")
torch_model.addDense(64, activation = "relu")
torch_model.addDense(32, activation="relu", dropout = 0.3)
torch_model.addDense(10, activation="softmax")

model = torch_model.get_model()

print(model)
print(table(torch_model.arch_summary()))
print(table(torch_model.param_summary()))

