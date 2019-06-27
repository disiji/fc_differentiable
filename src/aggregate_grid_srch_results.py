import os
import numpy as np


def is_grid_srch_res(path_to_dir, directory):
    return os.path.isdir(os.path.join(path_to_dir, directory)) and (directory.split('_')[0] == 'grid')

def aggregate_grid_srch_res(res_dir):
    
    dirs = [d for d in os.listdir(res_dir) if is_grid_srch_res(res_dir, d)]
    reg_settings = []
    unfinished = []
    results = []
    for d in dirs:
        if len(os.listdir(os.path.join(res_dir, d))) < 3:
            unfinished.append(d)
            continue
        split_name = d.split('_')
        neg_box_setting = float(split_name[3][4:])
        corner_setting = float(split_name[4][7:])
        gate_size_setting = float(split_name[6][5:])
        reg_settings.append((neg_box_setting, corner_setting, gate_size_setting))
        
        with open(os.path.join(res_dir, d, 'results_cll_4D.csv'), 'rb') as f:
            res = np.genfromtxt(f, delimiter=',')
            acc = float(res[1][1])
            overlap = float(res[1][-2])
            results.append((acc, overlap))

    with open(os.path.join(res_dir, 'agg_res.csv'), 'w') as f:
        f.write('neg_box, corner, gate_size, acc, dafi_overlap\n')
        for reg_setting, res in zip(reg_settings, results):
            f.write('%.1f, %.1f, %.1f, ' %reg_setting)
            f.write('%.2f, %.2f\n' %res)

    print(unfinished)
        
        
if __name__ == '__main__':
    results_dir = '../output/'
    aggregate_grid_srch_res(results_dir)
