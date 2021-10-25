try:
    import sys
    import copy
    import time
    from numpy.lib.function_base import average
    import torch
    import warnings
    from tqdm import tqdm
    import torch.nn as nn
    import torch.optim as optim
    from Training.metrics import TorchMetrics
    print("=====> (Train Eval Module) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f"ERROR: {e} Install modules properly ....")

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'


class TorchTrain:
    def __init__(self, data_loaders, model, criterion, optimizer,
                 schedular, num_epochs, task, save_checkpoints_path,
                 early_stop, show_batch_logs=True):
        """
        params:
        ------
        data_loaders : dict
                       It will contain two phase one is for training and other is for the validation
                       e.g. train_loader -> data_loaders['train']
                            valid_loader -> data_loaders['valid']
        
        model : torch state dict / torchvision.models
                The pytorch model, which can be transfer learning model, or a custom model
        
        criterion : loss function
                    Defining the loss function for the back propagation with torch autograd

        optimizer : torch optimizer
                    Defining the optimize the models weight
        
        num_epochs : int
                     The number of the epochs the model will be trained
        
        task : string
                This will take the correct type of metrics on the basis of the taks
                i.e. for the classification purpose, it will consider metrics like 
                1. loss
                2. accuracy
                3. precision
                4. recall
                5. f1
                Where as for regression type of tasks, it will choose metrics like
                1. loss
                2. r2-score 
                etc
        
        save_checkpoints_path : string
                                It will save the model's state dict on the given path (if not None)
        
        early_stop : boolean
                     Whether to enable for early stop or not
        
        show_batch_logs : boolean
                          Whether to show the logs in the batches

        """
        self.dataloaders = data_loaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = num_epochs
        self.task = task
        self.save_checkpoints_path = save_checkpoints_path
        self.early_stop_after = early_stop
        self.show_batch_logs = show_batch_logs

        if self.task == 'classification':
            self.history = {
                'train_loss': [],
                'train_acc': [],
                'train_precision': [],
                'train_recall': [],
                'train_f1': [],

                'valid_loss': [],
                'valid_acc': [],
                'valid_precision': [],
                'valid_recall': [],
                'valid_f1': []
            }
        else:
            self.history = {
                'train_loss': [],
                'train_r2': [],

                'valid_loss': [],
                'valid_r2': []
            }

        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def train_model(self):
        start_time = time.time()
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

        best_acc = 0.0
        lowest_loss = float('inf')
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        stop_count = 0
        best_r2 = 0.0

        print('Training started ....')
        batch_count = next(iter(self.dataloaders['train']))[0].shape[0]

        for epoch in range(self.num_epochs):
            print()
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            epoch_acc = 0.0
            epoch_loss = float('inf')
            epoch_precision = 0.0
            epoch_recall = 0.0
            epoch_f1 = 0.0

            if self.task != 'classification':
                epoch_r2 = 0.0

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_correct = 0.0
                running_precision = 0.0
                running_recall = 0.0
                running_f1 = 0.0
                running_r2 = 0.0

                batches_till = 0.0
                for inputs, labels in self.dataloaders[phase]:
                    batches_till += 1
                    X = inputs.to(self.device)
                    y = labels.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(X)
                        confidence, predictions = torch.max(outputs, 1)
                        loss = self.criterion(outputs, y)
                        batch_metrics = TorchMetrics(predictions, y)

                        if self.task == 'classification':
                            accuracy = batch_metrics.accuracy()
                            precision = batch_metrics.get_precision_score()
                            recall = batch_metrics.get_recall_score()
                            f1_score = batch_metrics.getf1_score()

                            sys.stdout.write("\r{}> {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                                "=" * int(batches_till / batch_count),
                                phase,
                                'after_batch:', batches_till,
                                'loss:', [loss.item()],
                                'acc:', [accuracy],
                                'precision:', [precision],
                                'recall:', [recall],
                                'f1:', [f1_score]))

                            sys.stdout.flush()
                            time.sleep(0.5)

                        else:
                            r2_score = batch_metrics.r2_score()
                            sys.stdout.write("\r{}> {} {} {} {} {} {} {}".format(
                                "=" * int(batches_till / batch_count),
                                phase, 'after_batch', batches_till,
                                'loss:', [loss.item()],
                                'r2-score:', [r2_score]))

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    if self.task == 'classification':
                        running_correct += torch.sum(predictions == y.data)
                        running_metrics = TorchMetrics(predictions, y)
                        running_precision += running_metrics.get_precision_score()
                        running_recall += running_metrics.get_recall_score()
                        running_f1 += running_metrics.getf1_score()
                    else:
                        running_r2 += running_metrics.r2_score()

                    if phase == 'train':
                        self.history['train_loss'].append(running_loss)
                        if self.task == 'classification':
                            self.history['train_acc'].append(running_correct / batches_till)
                            self.history['train_precision'].append(running_precision / batches_till)
                            self.history['train_recall'].append(running_recall / batches_till)
                            self.history['train_f1'].append(running_f1)
                        else:
                            self.history['train_r2'].append()

                    if phase == 'valid':
                        if self.task == 'classification':
                            self.history['valid_acc'].append(running_correct / batches_till)
                            self.history['valid_precision'].append(running_precision / batches_till)
                            self.history['valid_recall'].append(running_recall / batches_till)
                            self.history['valid_f1'].append(running_f1)
                        else:
                            self.history['valid_r2'].append(running_r2)

                if phase == 'train':
                    self.schedular.step()

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)

                if self.task == 'classification':
                    epoch_acc = running_correct / len(self.dataloaders[phase].dataset)
                    epoch_precision = running_precision / batches_till
                    epoch_recall = running_recall / batches_till
                    epoch_f1 = running_f1 / batches_till
                else:
                    epoch_r2 = running_r2 / batches_till
                print()

                if self.task == 'classification':
                    print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-score: {:.4f}'.format(
                        phase,
                        epoch_loss,
                        epoch_acc,
                        epoch_precision,
                        epoch_recall,
                        epoch_f1
                    ))
                else:
                    print('{} Loss: {:.4f} r2-score: {:.4f}'.format(
                        phase,
                        epoch_loss,
                        epoch_r2
                    ))

                if phase == 'valid':
                    if self.task == 'classification':
                        average_best_metrics = (best_acc + lowest_loss + best_precision + best_recall + best_f1) / 5
                        average_epoch_metrics = (epoch_acc + epoch_loss + epoch_precision + epoch_recall + epoch_f1) / 5
                    else:
                        average_best_metrics = (lowest_loss + best_r2)
                        average_epoch_metrics = (epoch_loss + epoch_r2)

                    if epoch > 0 and average_epoch_metrics > average_best_metrics:
                        print('Saving best weights of the model ...')
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

                    if self.task == 'classification':
                        best_acc = max(epoch_acc, best_acc)
                        lowest_loss = min(lowest_loss, epoch_loss)
                        best_precision = max(best_precision, epoch_precision)
                        best_recall = max(best_recall, epoch_recall)
                        best_f1 = max(best_f1, epoch_f1)
                    else:
                        best_r2 = max(best_r2, epoch_r2)

                    if self.task == 'classification':
                        if epoch > 0 and epoch_loss > lowest_loss and epoch_acc < best_acc:
                            stop_count += 1
                        else:
                            stop_count = 0

                    else:
                        if epoch > 0 and epoch_loss > lowest_loss and epoch_r2 < best_r2:
                            stop_count += 1
                        else:
                            stop_count = 0

                    print()
                    if stop_count >= self.early_stop_after:
                        print(f'Early stopping after epoch {epoch} ...')
                        break

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val Loss: {:4f}'.format(lowest_loss))
        print('Best val Precision: {:4f}'.format(best_precision))
        print('Best val Precision: {:4f}'.format(best_recall))
        print('Best val Precision: {:4f}'.format(best_f1))

        self.model.load_state_dict(self.best_model_wts)
        return self.model
