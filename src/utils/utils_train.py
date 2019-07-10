from copy import deepcopy

from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


class Tracker():
    def __init__(self):
        # Keep track of losses for plotting
        self.loss = []
        self.log_loss = []
        self.ref_reg_loss = []
        self.size_reg_loss = []
        self.corner_reg_loss = []
        self.acc = []
        self.precision = []
        self.recall = []
        self.roc_auc_score = []
        self.brier_score_loss = []
        self.log_decision_boundary = []
        self.modle_opt = None
        self.model_init = None
        self.acc_opt = 0
        self.n_iter_opt = (0, 0)

    def update(self, model, output, y_true, epoch, i, update_type='lightweight'):
        y_pred = (output['y_pred'].cpu().detach().numpy() > 0.5) * 1.0
        self.loss.append(output['loss'])
        self.log_loss.append(output['log_loss'])
        self.ref_reg_loss.append(output['ref_reg_loss'])
        self.size_reg_loss.append(output['size_reg_loss'])
        self.corner_reg_loss.append(output['corner_reg_loss'])
        self.acc.append(sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0])
        if not(update_type== 'lightweight'):
            
            self.precision.append(precision_score(y_true.cpu().numpy(), y_pred, average='macro'))
            self.recall.append(recall_score(y_true.cpu().numpy(), y_pred, average='macro'))
        #removed to see if it gives a speed improvement
        #self.roc_auc_score.append(roc_auc_score(y_true.numpy(), y_pred, average='macro')) 
            self.brier_score_loss.append(brier_score_loss(y_true.cpu().numpy(), output['y_pred'].cpu().detach().numpy()))
            self.log_decision_boundary.append(
            (-model.linear.bias.cpu().detach() / model.linear.weight.cpu().detach()))
        # keep track of optimal gates for train and eval set
            if self.acc[-1] > self.acc_opt:
                self.modle_opt = deepcopy(model)
                self.acc_opt = self.acc[-1]
                self.n_iter_opt = (epoch, i)
