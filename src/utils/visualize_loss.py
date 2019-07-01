import sys
sys.path.append('../')
import pickle
import numpy as np
import utils.utils_load_data as dh
import torch
from utils.bayes_gate import *
import matplotlib.pyplot as plt
from train import run_train_only_logistic_regression

X_DEV_PATH = '../../data/cll/x_dev_4d_1p.pkl'
Y_DEV_PATH = '../../data/cll/y_dev_4d_1p.pkl'
COLORS = ['r', 'b', 'g', 'y', 'm', 'k']
#for plotting with only the use of the CD19/CD5 and CD79b/CD10 gates
def visualize_loss_as_gates_move_1p_leaf(leaf_gate_init, DAFI_leaf_gate, DAFI_root_gate, num_steps, make_gif=False, x_data_dir=X_DEV_PATH, y_data_dir=Y_DEV_PATH, figsize=(5, 12)):
    x_list, y_list = load_data(x_data_dir, y_data_dir)
    x_list, offset, scale = dh.normalize_x_list(x_list)
    x_list = [torch.tensor(_, dtype=torch.float32) for _ in x_list]

    y_list = torch.tensor(y_list, dtype=torch.float32)

    #TODO make the entire function take in a params dict for the reg_weights
    log_reg_lr = .03 #3e-4

    model, dafi_tree = setup_model_and_dafi_tree(DAFI_root_gate, DAFI_leaf_gate, leaf_gate_init, offset, scale)
    print('Training model for the first time')
    model = run_train_only_logistic_regression(model, x_list, y_list, log_reg_lr, verbose=False)
    init_output = model.forward(x_list, y_list)
    y_pred = (init_output['y_pred'].detach().numpy() > 0.5) * 1.0
    init_output = {item:init_output[item].detach().item() for item in init_output if item in ['log_loss', 'ref_reg_loss', 'emp_reg_loss', 'corner_reg_loss', 'size_reg_loss']}

    log_losses = [init_output['log_loss']]
    ref_regs = [init_output['ref_reg_loss']]
    gate_size_regs = [init_output['size_reg_loss']]
    neg_regs = [init_output['emp_reg_loss']]
    corner_regs = [init_output['corner_reg_loss']]
    accs = [sum(y_pred == y_list.detach().numpy()) * 1.0 / y_list.shape[0]]

    #compute the line connecting the center of the rectangles
    leaf_gate_center = [(leaf_gate_init[0] + leaf_gate_init[1])/2, (leaf_gate_init[2] + leaf_gate_init[3])/2]
    DAFI_leaf_gate_center= [(DAFI_leaf_gate[0] + DAFI_leaf_gate[1])/2, (DAFI_leaf_gate[2] + DAFI_leaf_gate[3])/2]

    distance = ((DAFI_leaf_gate_center[1] - leaf_gate_center[1])**2 + (DAFI_leaf_gate_center[0] - leaf_gate_center[0])**2)**(1/2)
    step_size = distance/num_steps
    slope_between = (DAFI_leaf_gate_center[1] - leaf_gate_center[1])/(DAFI_leaf_gate_center[0] - leaf_gate_center[0])
    unit_vector = 1./(np.sqrt(slope_between**2 + 1)) * np.array([1, slope_between])
    
    gate_fig, gate_ax = plt.subplots(1)

    for s in range(1, num_steps + 1):
        print('Step %d/%d' %(s, num_steps))
        step = -unit_vector * s *  step_size
        leaf_gate = [leaf_gate_init[0] + step[0], leaf_gate_init[1] + step[0], leaf_gate_init[2] + step[1], leaf_gate_init[3] + step[1]]
        plot_gate(gate_ax, leaf_gate, COLORS[s%len(COLORS)], 'Moving Leaf Gate')
        cur_model =  ModelTree(dafi_tree, init_tree=get_tree_1p(DAFI_root_gate, leaf_gate, offset, scale))
        cur_model = run_train_only_logistic_regression(cur_model, x_list, y_list, log_reg_lr, verbose=False)
        cur_out = cur_model.forward(x_list, y_list)
        y_pred = (cur_out['y_pred'].detach().numpy() > 0.5) * 1.0
        accs.append(sum(y_pred == y_list.detach().numpy()) * 1.0 / y_list.shape[0])
        #print(cur_model)
        cur_out = {item:cur_out[item].detach().item() for item in cur_out if item in ['log_loss', 'ref_reg_loss', 'emp_reg_loss', 'corner_reg_loss', 'size_reg_loss']}
 
        log_losses.append(cur_out['log_loss'])
        ref_regs.append(cur_out['ref_reg_loss'])
        gate_size_regs.append(cur_out['size_reg_loss'])
        neg_regs.append(cur_out['emp_reg_loss'])
        corner_regs.append(cur_out['corner_reg_loss'])

    #now make plots
    distances = [distance - step_size*n for n in range(num_steps)]
    distances.append(0)
    print(distances)
    print(log_losses)
    
    fig, axes = plt.subplots(6, 1, figsize=figsize)
    axes[0].set_title('Distance Between Centers vs log_loss')
    axes[1].set_title('Distance Between Centers vs ref_regs')
    axes[2].set_title('Distance Between Centers vs gate_size_regs')
    axes[3].set_title('Distance Between Centers vs neg_regs')
    axes[4].set_title('Distance Between Centers vs corner_regs')
    axes[5].set_title('Distance Between Centers vs accuracy')
    
    axes[0].plot(distances, log_losses)
    axes[1].plot(distances, ref_regs)
    axes[2].plot(distances, gate_size_regs)
    axes[3].plot(distances, neg_regs)
    axes[4].plot(distances, corner_regs)
    axes[5].plot(distances, accs)
    fig.tight_layout() 
    fig.savefig('../../output/cll_4d_1p_loss_moving_gate_between.png')
    gate_fig.savefig('../../output/debug_gate_motion.png')

        
        
        

def load_data(x_data_dir, y_data_dir): 
    with open(x_data_dir, 'rb') as f:
        x_list = pickle.load(f)
    with open(y_data_dir, 'rb') as f:
        y_list = pickle.load(f)    
    #x_list = [torch.tensor(_, dtype=torch.float32) for _ in x_list]
    #y_list = torch.tensor(y_list, dtype=torch.float32)
    return x_list, y_list

def setup_model_and_dafi_tree(DAFI_root_gate, DAFI_leaf_gate, leaf_gate_init, offset, scale): 
    init_tree = get_tree_1p(DAFI_root_gate, leaf_gate_init, offset, scale)
    dafi_tree = get_tree_1p(DAFI_root_gate, DAFI_leaf_gate, offset, scale)
    
    model = ModelTree(dafi_tree, init_tree=init_tree)
    return model, dafi_tree

def get_tree_1p(root_gate, leaf_gate, offset, scale):
    
    features = ['CD5', 'CD19', 'CD10', 'CD79b']
    feature2id = dict((features[i], i) for i in range(len(features)))
    nested_list = \
            [
                [[u'CD5', root_gate[0], root_gate[1]], [u'CD19', root_gate[2], root_gate[3]]],
                [
                    [
                        [[u'CD10', leaf_gate[0], leaf_gate[1]], [u'CD79b', leaf_gate[2], leaf_gate[3]]],
                        []
                    ]
                ]
            ]

    nested_list = dh.normalize_nested_tree(nested_list, offset, scale, feature2id)
    tree = ReferenceTree(nested_list, feature2id)

    return tree

def plot_box(axes, x1, x2, y1, y2, color, label, dashed=False, lw=3):
    dash = [3,1]
    if dashed:
        axes.plot([x1, x1], [y1, y2], c=color, label=label, dashes=dash, linewidth=lw)
        axes.plot([x1, x2], [y1, y1], c=color, dashes=dash, linewidth=lw)
        axes.plot([x2, x2], [y1, y2], c=color, dashes=dash, linewidth=lw)
        axes.plot([x2, x1], [y2,y2], c=color, dashes=dash, linewidth=lw)
    else:
        axes.plot([x1, x1], [y1, y2], c=color, label=label, linewidth=lw)
        axes.plot([x1, x2], [y1, y1], c=color, linewidth=lw)
        axes.plot([x2, x2], [y1, y2], c=color, linewidth=lw)
        axes.plot([x2, x1], [y2,y2], c=color, linewidth=lw)
    return axes

def plot_gate(axes, gate, color, label, dashed=False, lw=.5):
    plot_box(axes, gate[0], gate[1], gate[2], gate[3], color, label, dashed=dashed, lw=lw)



if __name__ == '__main__':


    center_leaf = [1019., 3056., 979., 2937.]
    similar_size_dafi_leaf = [1400., 2600., 1100, 2900]
    num_steps = 50
    visualize_loss_as_gates_move_1p_leaf(similar_size_dafi_leaf, [0., 1228., 0., 1843.], [1638., 3891., 2150., 3891.], num_steps)
