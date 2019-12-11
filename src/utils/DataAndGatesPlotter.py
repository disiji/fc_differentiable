import seaborn as sb

import utils.utils_load_data as dh
from utils.bayes_gate import ModelTree


class DataAndGatesPlotter():
    '''
    Class to handle plotting the data and gates

    attributes:
        model: the model whose gates to plot
        data: the data to plot
        filtered_data: data filtered along the tree structure 
                        defined by model in depth first order
        dims: the indexes into self.data for which two dimensions
                a given node uses
    '''

    def __init__(self, model, data, is_4chain=False, color=None):
        self.model = model
        self.data = data
        self.gates = self.construct_gates()
        self.filtered_data = self.filter_data_by_model_gates()
        self.dims = self.get_dims()
        self.color = color


    '''
    loads the nodes gate params into a namedtuple
    '''

    @staticmethod
    def get_gate(node):
        gate = ModelTree.get_gate(node)
        return gate

    @staticmethod
    def get_dim_single_node(node):
        return (node.gate_dim1, node.gate_dim2)

    def get_dims(self):
        dims, _ = self.apply_function_depth_first(
            DataAndGatesPlotter.get_dim_single_node
        )
        return dims

    '''
    filters data using the gate from the input node
    '''

    def filter_data_at_single_node(self, data, node):
        gate = DataAndGatesPlotter.get_gate(node)
        filtered_data = dh.filter_rectangle(
            data, node.gate_dim1,
            node.gate_dim2, gate.low1, gate.upp1,
            gate.low2, gate.upp2
        )
        return filtered_data

    def construct_gates(self):
        return self.apply_function_depth_first(
            DataAndGatesPlotter.get_gate
        )[0]

    def filter_data_by_model_gates(self):
        # Pass a dummy function to just compute filtered_data
        _, filtered_data = self.apply_function_depth_first(lambda x, y: None,
                                                           function_uses_data=True)
        return filtered_data

    '''
    applies the given function to each node in 
    the model tree in a depth first order, currently only
    implemented for a chain/line graph

    param function: the function to apply at each node
    
    returns output: A list of results from applying the function
                    to each node in depth first order.

    returns filtered_data: The filtered data at each node in depth first
                           order
    '''

    def apply_function_depth_first(self, function, function_uses_data=False):
        # lists easily function as stacks in python
        node_stack = [self.model.root]
        if function_uses_data:
            # keep track of each's node parent data after filtering
            data_stack = [self.data]

        filtered_data = [self.data]
        outputs = []

        while len(node_stack) > 0:
            node = node_stack.pop()

            if function_uses_data:
                data = data_stack.pop()

                # call function on filtered data from the node's parent
                outputs.append(function(node, filtered_data[-1]))

                filtered_data.append(self.filter_data_at_single_node(data, node))
            else:
                outputs.append(function(node))

            for child in self.model.children_dict[self.model.get_node_idx(node)]:
                node_stack.append(child)
                if function_uses_data:
                    # push the same data onto the stack since the
                    # children share the same parent
                    data_stack.append(filtered_data[-1])

        return outputs, filtered_data

    '''
    plots on an 1-d np array of axes the filtered data 
    and the gates for each node
    '''

    def plot_on_axes(self, axes, hparams):

        if not (axes.shape[0] == len(self.filtered_data) - 1):
            print(self.gates)
            print(len(self.filtered_data))
            raise ValueError('Number of axes must match the number of nodes!')

        for node_idx, axis in enumerate(axes):
            self.plot_node(axis, node_idx, hparams)

    def plot_node(self, axis, node_idx, hparams):
        if 'plot_kde_density' in hparams['plot_params']:
            if hparams['plot_params']['plot_kde_density']:
                sb.kdeplot(
                    self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                    self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                    ax=axis, cmap='Blues', shade=True, shade_lowest=False
                )
        else:
            axis.scatter(
                self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                s=hparams['plot_params']['marker_size'],
            )
        if self.color is None:
            self.plot_gate(axis, node_idx, dashes=(3, 1), label='Model')
        else:
            self.plot_gate(axis, node_idx, color=self.color, label='DAFI')

    def plot_only_DAFI_gates_on_axes(self, axes, hparams):
        if not (axes.shape[0] == len(self.filtered_data) - 1):
            raise ValueError('Number of axes must match the number of nodes!')
        for node_idx, axis in enumerate(axes):
            self.plot_gate(axis, node_idx, color='k', label='DAFI')

    def plot_gate(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color,
                  label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color,
                  dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color,
                  dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2, gate.upp2], c=color,
                  dashes=dashes, linewidth=lw)
        return axis


class DataAndGatesPlotterBoth(DataAndGatesPlotter):
    '''
    Class to handle plotting the data and gates

    attributes:
        model: the model whose gates to plot
        data: the data to plot
        filtered_data: data filtered along the tree structure 
                        defined by model in depth first order
        dims: the indexes into self.data for which two dimensions
                a given node uses
    '''

    def __init__(self, model, data_both, is_4chain=False, color=None):
        self.model = model
        self.data = data_both
        self.nodes = self.get_nodes()
        self.gates = self.construct_gates()
        self.filtered_data = self.filter_data_by_model_gates(data_both)
        self.dims = self.get_dims()
        self.color = color


    def get_dims(self):
        dims = []
        for node in self.nodes:
            dims.append([node.gate_dim1, node.gate_dim2])
        return dims

    def get_nodes(self):
        return self.model.get_flat_nodes()

    def construct_gates(self):
        gates = []
        for node in self.nodes:
            gates.append(ModelTree.get_gate(node))
        return gates

    def filter_data_by_model_gates(self, data_both):
        return self.model.get_filtered_data_all_nodes(data_both)

    '''
    plots on an 1-d np array of axes the filtered data 
    and the gates for each node
    '''

    def plot_on_axes(self, axes, hparams):

        if not (axes.shape[0] == len(self.filtered_data)):
            print(self.gates)
            print(len(self.filtered_data))
            raise ValueError('Number of axes must match the number of nodes!')

        for node_idx, axis in enumerate(axes):
            self.plot_node(axis, node_idx, hparams)

    def plot_node(self, axis, node_idx, hparams):
        if 'plot_kde_density' in hparams['plot_params']:
            if hparams['plot_params']['plot_kde_density']:
                sb.kdeplot(
                    self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                    self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                    ax=axis, cmap='Blues', shade=True, shade_lowest=False
                )
        else:
            axis.scatter(
                self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                s=hparams['plot_params']['marker_size'],
            )
        if self.color is None:
            self.plot_gate(axis, node_idx, dashes=(3, 1), label='Model')
        else:
            self.plot_gate(axis, node_idx, color=self.color, label='DAFI')

    def plot_only_DAFI_gates_on_axes(self, axes, hparams):
        if not (axes.shape[0] == len(self.filtered_data) - 1):
            raise ValueError('Number of axes must match the number of nodes!')
        for node_idx, axis in enumerate(axes):
            self.plot_gate(axis, node_idx, color='k', label='DAFI')

    def plot_gate(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color,
                  label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color,
                  dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color,
                  dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2, gate.upp2], c=color,
                  dashes=dashes, linewidth=lw)
        return axis
