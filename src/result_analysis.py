import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

column_names = ['random_state',
                'train_accuracy',
                'eval_accuracy',
                'overall_accuracy',
                'train_accuracy_dafi',
                'eval_accuracy_dafi',
                'overall_accuracy_dafi',
                'train_tracker.acc_opt'
                'eval_tracker.acc_opt',
                'train_logloss',
                'eval_logloss',
                'overall_logloss',
                'train_logloss_dafi',
                'eval_logloss_dafi',
                'overall_logloss_dafi',
                'train_auc',
                'eval_auc',
                'overall_auc',
                'train_auc_dafi',
                'eval_auc_dafi',
                'overall_auc_dafi',
                'train_brier_score',
                'eval_brier_score',
                'overall_brier_score',
                'train_brier_score_dafi',
                'eval_brier_score_dafi',
                'overall_brier_score_dafi',
                'run_time']

#  generate scatter plots of a method and dafi gates
metric_dict = ['accuracy', 'logloss', 'auc', 'brier_score']
# model_dict = ['default', 'dafi_init', 'dafi_regularization', 'default_non_alternate', 'emp_regularization_off',
#               'gate_size_regularization_off']
model_dict = ['default']


# load from csv
def scatter_vs_dafi_feature(dataname, method_name, metric_name, ax):
    filename = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df = pd.read_csv(filename, header=None, names=column_names)
    # stats test
    print(stats.ttest_rel(df['eval_%s' % metric_name],df['eval_%s_dafi' % metric_name]))
    print(stats.ks_2samp(df['eval_%s' % metric_name],df['eval_%s_dafi' % metric_name]))

    ax.scatter(df['eval_%s' % metric_name], df['eval_%s_dafi' % metric_name], s=5)
    ax.set_xlim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])),\
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.set_ylim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])),\
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    return ax


def scatter_methods(dataname, method_name_1, method_name_2, metric_name):
    # todo: need to make df of different methods to plot of same length
    """

    :param dataname:
    :param method_name_1:
    :param method_name_2:
    :param metric_name:
    :return:
    """
    filename_1 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_1)
    filename_2 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_2)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df_1 = pd.read_csv(filename_1, header=None, names=column_names)
    df_2 = pd.read_csv(filename_2, header=None, names=column_names)

    figname_train = '../fig/%s/%s_vs_%s_%s_train.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['train_%s' % metric_name], df_2['train_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_train)

    figname_test = '../fig/%s/%s_vs_%s_%s_test.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['eval_%s' % metric_name], df_2['eval_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.plot()
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_test)


if __name__ == '__main__':
    dataname = 'cll_4d_1p'
    for method_name in model_dict:
        figname = '../output/%s/comparison.pdf' % (dataname + '_' + method_name)
        f, axarr = plt.subplots(1, len(metric_dict), figsize=(10, 2))
        for i, metric_name in enumerate(metric_dict):
            axarr[i] = scatter_vs_dafi_feature(dataname, method_name, metric_name, axarr[i])
            axarr[i].set_xlabel('Model gates')
        axarr[0].set_ylabel('Expert gates')
        f.savefig(figname, bbox_inches='tight')
    # for i in range(len(model_dict)):
    #     for j in range(i + 1, len(model_dict)):
    #         for metric_name in metric_dict:
    #             scatter_methods(dataname, model_dict[i], model_dict[j], metric_name)
