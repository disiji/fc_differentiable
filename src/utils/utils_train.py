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
        self.acc = []
        self.precision = []
        self.recall = []
        self.roc_auc_score = []
        self.brier_score_loss = []
        self.log_decision_boundary = []
        self.root_gate_opt = None
        self.leaf_gate_opt = None
        self.root_gate_init = None
        self.leaf_gate_init = None
        self.acc_opt = 0
        self.n_iter_opt = (0, 0)

    def update(self, model, output, y_true, epoch, i):
        y_pred = (output['y_pred'].detach().numpy() > 0.5) * 1.0
        self.loss.append(output['loss'])
        self.log_loss.append(output['log_loss'])
        self.ref_reg_loss.append(output['ref_reg_loss'])
        self.size_reg_loss.append(output['size_reg_loss'])
        self.acc.append(sum(y_pred == y_true.numpy()) * 1.0 / y_true.shape[0])
        self.precision.append(precision_score(y_true.numpy(), y_pred, average='macro'))
        self.recall.append(recall_score(y_true.numpy(), y_pred, average='macro'))
        self.roc_auc_score.append(roc_auc_score(y_true.numpy(), y_pred, average='macro'))
        self.brier_score_loss.append(brier_score_loss(y_true.numpy(), y_pred))
        self.log_decision_boundary.append(
            (-model.linear.bias.detach() / model.linear.weight.detach()))
        # keep track of optimal gates for train and eval set
        if self.acc[-1] > self.acc_opt:
            self.root_gate_opt = deepcopy(model.root)
            self.leaf_gate_opt = deepcopy(model.children_dict[str(id(model.root))][0])
            self.acc_opt = self.acc[-1]
            self.n_iter_opt = (epoch, i)