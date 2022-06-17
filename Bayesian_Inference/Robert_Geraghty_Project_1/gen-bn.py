#!/usr/bin/env python3

import json
import itertools
import numpy as np


def gen_bn(network_type, num_nodes, prob=0.5, max_parents=None, a=1.0, b=1.0):
    if network_type == 'dag':
        net = gen_dag(num_nodes, prob, max_parents)
    elif network_type == 'polytree':
        net = gen_polytree(num_nodes, prob, max_parents)
    else:
        raise Exception("Invalid network type. "
                        "Valid types are 'dag' and 'polytree'.")
    bn = {i:{} for i in range(num_nodes)}
    for i, p in net.items():
        bn[i]['parents'] = sorted(p)
        bn[i]['prob'] = {}
        for assn in itertools.product(*([(0,1)] * len(p))):
            bn[i]['prob'][assn] = np.random.beta(a, b)
        bn[i]['prob'] = list(bn[i]['prob'].items())
    return bn


def gen_dag(num_nodes, prob=0.5, max_parents=None):
    net = {}
    for n in range(num_nodes):
        ps = n + 1 + np.argwhere(np.random.uniform(size=num_nodes-n-1) > 0.5)
        ps = ps.T.tolist()[0]
        if max_parents:
            np.random.shuffle(ps)
            ps = ps[:max_parents]
        net[n] = ps
    return net


def gen_polytree(num_nodes, prob=0.5, max_parents=None):
    clusters = []
    net = {}
    for i in range(num_nodes):
        net[i] = []
        nc = [[i]]
        new_clusters = []
        for c in clusters:
            if np.random.uniform() < prob:
                nc += [c]
                j = int(np.random.choice(c))
                net[i] += [j]
            else:
                new_clusters += [c]
            if max_parents and len(net[i]) >= max_parents:
                break
        nc = list(itertools.chain(*nc))
        new_clusters += [nc]
        clusters = new_clusters
    return net


def draw_net(net):
    import networkx as nx
    import matplotlib.pyplot as plt

    g = nx.DiGraph()

    ## add nodes
    for i in net.keys():
        g.add_node(i)

    ## add edges
    for i, data in net.items():
        for j in data['parents']:
            g.add_edge(i, j)


    nx.draw(g, with_labels=True)
    plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Generates random probabilistic "
                                        "networks with Bernoulli random "
                                        "variables.")
    parser.add_argument('network_type', type=str,
                        help="The network type to generate. Acceptable values "
                             "are 'dag' and 'polytree'.")
    parser.add_argument('num_nodes', type=int,
                        help="The number of nodes for the network")
    parser.add_argument('--prob', type=float, default=0.5,
                        help="The probability of generating an edge. "
                             "(default: 0.5)")
    parser.add_argument('--max-parents', type=int, default=5,
                        help="The maximum number of parents a node may have. "
                             "If you do not want to restrict the number of "
                             "parents, use -1. (default: 5)")
    parser.add_argument('--a', type=float, default=1.0,
                        help="The first hyperparameter to the beta "
                             "distribution that generate conditional "
                             "probabilities. (default: 1.0)")
    parser.add_argument('--b', type=float, default=1.0,
                        help="The second hyperparameter to the beta "
                             "distribution that generates conditional "
                             "probabilities. (default: 1.0)")
    parser.add_argument('--output-file', type=str, default='bn.json',
                        help="The JSON-formatted output file to write to.")
    parser.add_argument('--draw-net', dest='draw_net', action='store_true',
                        help="Creates a visualization of the network. "
                             "Requires packages `networkx` and `matplotlib`. "
                             "(default: False)")
    parser.set_defaults(draw_net=False)

    args = parser.parse_args()
    bn = gen_bn(args.network_type, args.num_nodes, args.prob,
                args.max_parents if args.max_parents > 0 else None,
                args.a, args.b)

    with open(args.output_file, 'w') as f:
        f.write(json.dumps(bn, indent=2))

    print("Written to '{}'.".format(args.output_file))
    if args.draw_net:
        draw_net(bn)
