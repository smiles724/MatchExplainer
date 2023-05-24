import argparse
from texttable import Texttable


def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())


def parse_args():
    parser = argparse.ArgumentParser(description="Match Explainer")

    ###################################################
    ##  Training
    ###################################################
    parser.add_argument('--cuda', type=int, default=0, help='GPU device.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of loops to train the mask.')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])
    parser.add_argument('--ratio', type=float, default=0.4)
    parser.add_argument('--hid', type=int, default=50, help='Hidden dim')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--data_root', type=str, default="data/")
    parser.add_argument('--param_root', type=str, default="param/")
    parser.add_argument('--log_root', type=str, default="log/")
    parser.add_argument('--recall', action='store_true', help="Test recall for BA3")
    parser.add_argument('--recall_k', type=int, default=5)

    return parser.parse_args()


def get_default_config():
    node_state_dim = 32
    edge_state_dim = 16
    graph_rep_dim = 128
    graph_embedding_net_config = dict(node_state_dim=node_state_dim, edge_state_dim=edge_state_dim,
                                      edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
                                      node_hidden_sizes=[node_state_dim * 2], n_prop_layers=5,
                                      # set to False to not share parameters across message passing layers
                                      share_prop_params=True,
                                      # initialize message MLP with small parameter weights to prevent
                                      # aggregated message vectors blowing up, alternatively we could also use
                                      # e.g. layer normalization to keep the scale of these under control.
                                      edge_net_init_scale=0.1,
                                      # other types of update like `mlp` and `residual` can also be used here. gru
                                      node_update_type='gru', use_reverse_direction=True,
                                      # set to True if your g is directed
                                      reverse_dir_param_different=False, layer_norm=False, prop_type='matching')
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # other: euclidean, cosine
    return dict(
        encoder=dict(node_hidden_sizes=[node_state_dim], node_feature_dim=1, edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(node_hidden_sizes=[graph_rep_dim], graph_transform_sizes=[graph_rep_dim],
                        input_size=[node_state_dim], gated=True, aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config, graph_matching_net=graph_matching_net_config,
        model_type='matching', data=dict(problem='graph_edit_distance',
                                         dataset_params=dict(n_nodes_range=[20, 20], p_edge_range=[0.2, 0.2],
                                                             n_changes_positive=1, n_changes_negative=2,
                                                             validation_dataset_size=1000)),
        training=dict(mode='pair', loss='margin', margin=1.0, graph_vec_regularizer_weight=1e-6, clip_value=10.0), seed=8)










