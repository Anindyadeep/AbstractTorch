try:
    import sys 
    from pathlib import Path
    BASE_DIR = "/home/anindya/Documents/pytorch/abt/ABT/AbstractTorch/v1" 
    sys.path.append(str(BASE_DIR))
    from Architectures import model_arch as ma 
    from Training import train_eval as te 
    import torch 
    import torch.nn as nn
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    print("=====> (Test Modules 1) modules imported successfully ....")
except ModuleNotFoundError as e: 
    print(f"ERROR: {e} Install modules properly ....")


train_data = MNIST(
            root = "/home/anindya/Documents/pytorch/abt/ABT/AbstractTorch/v1/tests/test_data",
            download = True,
            train = True,
            transform = ToTensor())

valid_data = MNIST(
            root = "/home/anindya/Documents/pytorch/abt/ABT/AbstractTorch/v1/tests/test_data",
            download = True,
            train = False,
            transform = ToTensor())

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = 64, shuffle = True)

torch_model = ma.TorchModel(train_loader, valid_loader)
torch_model.addConv2d(64, (3,3))
torch_model.addConv2d(128, (3,3), activation="relu")
torch_model.addConv2d(128, (3,3), activation="relu")
torch_model.addDense(256, activation = "relu")
torch_model.addDense(64, activation="relu")
torch_model.addDense(10)

print(torch_model.arch_summary(), "\n")
print(torch_model.param_summary(), "\n")

model = torch_model.get_model()

data_loaders = {"train": train_loader, "valid": valid_loader}
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
num_epochs = 2
task = "classification"
save_checkpts_path = "/home/anindya/Documents/pytorch/abt/ABT/AbstractTorch/v1/tests/chckpts"
early_stop = 1

tt = te.TorchTrain(
    data_loaders=data_loaders, 
    model = model, 
    criterion=criterion, 
    optimizer=optimizer, 
    schedular=scheduler, 
    num_epochs=num_epochs, 
    task=task, save_checkpoints_path=save_checkpts_path, early_stop=early_stop)

model = tt.train_model()