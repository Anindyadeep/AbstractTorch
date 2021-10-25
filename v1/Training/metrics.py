try:
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
                                confusion_matrix, 
                                precision_recall_curve, 
                                precision_score, 
                                recall_score, f1_score, 
                                fbeta_score, 
                                r2_score)
    print("=====> (Training Metrics) modules imported successfully ....")
except ModuleNotFoundError as e:
    print(f"ERROR: {e} Install modules properly ....")


class TorchMetrics:
    def __init__(self, preds, y):
        if type(y) == 'list':
            self.y = y
        else:
            self.y = y.tolist()
            self.num_classes = len(set(self.y))
        
        if type(preds) == 'list':
            self.preds = preds
        else:
            self.preds = preds.tolist()
        self.num_classes = len(self.y)

    def accuracy(self):
        num_correct =  (np.array(self.preds) == np.array(self.y)).sum()
        return num_correct/len(self.y)

    def getf1_score(self, average='micro'):
        if self.num_classes <= 2:
            return f1_score(self.y, self.preds)
        else:
            return f1_score(self.y, self.preds, average=average)

    def get_precision_score(self, average="micro"):
        if self.num_classes <= 2:
            return precision_score(self.y, self.preds)
        else:
            return precision_score(self.y, self.preds, average=average)

    def get_recall_score(self, average="micro"):
        if self.num_classes <= 2:
            return recall_score(self.y, self.preds)
        else:
            return recall_score(self.y, self.preds, average=average)

    def f_beta(self, beta, average):
        if self.num_classes <= 2:
            return fbeta_score(self.y, self.preds, beta)
        else:
            return fbeta_score(self.y, self.preds, beta=beta, average=average)

    def get_confusion_matrix(self, visualise=False, labels=None):
        cm = confusion_matrix(self.y, self.preds)
        if visualise:
            if labels:
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            else:
                cm_df = pd.DataFrame(cm)

            plt.figure(figsize=(5,4))
            sns.heatmap(cm_df, annot=True)
            plt.title('Confusion Matrix')
            plt.ylabel('Actal Values')
            plt.xlabel('Predicted Values')
            plt.show()
    
    def get_precision_recall_curve(self):
        precision = dict()
        recall = dict()
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y[:,i], self.preds[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        plt.show()

    def r2_score(self):
        return r2_score(self.y, self.preds)