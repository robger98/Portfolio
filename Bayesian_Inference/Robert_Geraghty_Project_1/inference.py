#!/usr/bin/env python

#
# Author: Robert Geraghty 
# Email : rrg053@utulsa.edu
# 

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph, write_dot, graphviz_layout
from networkx.algorithms.dag import topological_sort
import json
import random
import copy
from multiprocessing import Process, Lock, Manager
import time
import os
import cProfile
import ujson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', nargs=1, required = True, help='The name of the json file from gen-bn')
parser.add_argument('-d', '--draw', action='store_true', help='Number of samples for sampling')
parser.add_argument('-o', '--output', nargs=1, required = False, help='Output graphic file location')
parser.add_argument('-q', '--query', nargs=1, default='0', help='The query variable name')
parser.add_argument('-m', '--multiprocessing', action='store_true', help='Enables multiprocessing, probably don\'t use this if your other methods don\'t use multiprocessing')
args = parser.parse_args()

# to_draw = args.draw
# outfileloc = ''
# if to_draw:
#     if args.output == None:
#         print("-o OUTPUT_FILE_NAME or --output OUTPUT_FILE_NAME are required" +
#         " if you use flag -d or --draw!")
#         exit()
#     else:
#         outfileloc = str(args.output[0])
    
jsonfileloc=str(args.file[0])

multi = args.multiprocessing

class Bayes_Net():
    def __init__(self):
        self.net = nx.DiGraph()
        self.nodes = {}

    def create_from_json(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
            for name, value in data.items():
                node = Bayes_Node(str(name), [str(i) for i in value['parents']], value['prob'])
                self.nodes.update({str(name): node})
                self.net.add_node(node.name, cpt = node.cpt, color='black')
                for parent in node.parents:
                    self.net.add_edge(parent, node.name, label=(parent+"->"+node.name), color='black')#, minlen=(abs(int(parent)-int(node.name))*1))
    def add_node(self, node):
        self.net.add_node(node.name, cpt = node.cpt)
        for parent in node.parents:
            self.net.add_edge(parent.name, node.name)
    
    def prob_util(self, var, evidence, prob):
        return 1-prob if evidence[var] == 0 else prob 

    def enumeration_ask(self, query_var, evidence = {}):
        manager = Manager()
        Q = manager.dict()
        possibilities = [0,1]
        if query_var in evidence:
            other_vals = [x for x in possibilities if x != evidence[query_var]]
            out = {evidence[query_var]:1}
            for val in other_vals:
                out.update({val:0})
            return out
        topsort = list(topological_sort(self.net))
        if not multi:
            for x in possibilities:
                print('Enumerating with query var value', x)
                e = evidence
                e.update({query_var  :x})
                Q[x] = self.enumerate_all(topsort, e)
            return self.normalize(Q)
        else:
            lock = Lock()
            processes = []
            for x in possibilities:
                print('Enumerating with query var value', x)
                e = json_deep_copy(evidence)
                e.update({query_var  :x})
                processes.append(Process(target=self.enumerate_all, args=(topsort, e, Q, lock, query_var)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            return self.normalize(Q)

    def enumerate_all(self, v, ev, Q = None, lock = None, query_var = None):
        evidence = None
        varlist = None
        if lock == None:
            evidence = json_deep_copy(ev)
            varlist = json_deep_copy(v)
        else:
            lock.acquire()
            evidence = json_deep_copy(ev)
            varlist = json_deep_copy(v)
            lock.release()
        if varlist == []:
            if lock == None:
                return 1.0
            lock.acquire()
            Q.update({evidence[query_var]:1.0})
            lock.release()
            return

        Y = varlist[0]
        if Y in evidence:
            prob = self.prob_util(Y, evidence, self.P_x_given(Y, evidence))
            ret = prob * self.enumerate_all(varlist[1:], evidence)
            #print("Probability of {} is {} given {} is {}".format(str(Y), str(evidence[Y]), str(evidence), str(ret)))
            if lock == None:
                return ret
            lock.acquire()
            Q.update({evidence[query_var]:ret})
            lock.release()
            return
        else:
            e = evidence
            sum = 0
            for val in [1,0]:
                e.update({Y: val})
                ret = self.prob_util(Y, e, self.P_x_given(Y, e)) * self.enumerate_all(varlist[1:], e)
                #print("Probability of {} is {} given {} is {}".format(str(Y), str(e[Y]), str(e), str(ret)))
                sum += ret
            if lock == None:
                return sum
            lock.acquire()
            Q.update({evidence[query_var]:sum})
            lock.release()
            return
   
    def P_x_given(self, x, evidence):
        parent_values = []
        for parent in self.net.predecessors(x):
           parent_values.append(evidence[parent])
           
        match = [cp for cp in self.nodes[x].cpt if cp[0] == parent_values]
        return match[0][1]
    
    def normalize(self, distribution):
        # print(distribution)
        s = sum(list(distribution.values()))
        for key in list(distribution.keys()):
            distribution.update({key:distribution[key]/s})
        return distribution

    def  gibbs_ask(self, target, n_iters, evidence={}):
        Z = [i for i in self.net.nodes if i not in evidence] # get the nonevidence variables
        N = {0: 0, 1:0} # Dictionary for number of counts for each value of x
        x = evidence # the current state of the network
        for var in Z:
            x[var] =  random.randint(0, 1)
        for _ in range(1, n_iters):
            # print()
            for Z_i in Z:
                sample = random.random()
                prob = self.prob_util(Z_i, x, self.P_x_given_mb(Z_i, x))
                # print("Var", Z_i, "Vals", x,  "Prob", prob)
                x[Z_i] = 1 if sample <= prob else 0
                N[x[target]] = N[x[target]] + 1
        return self.normalize(N)
       
    
    def P_x_given_mb(self, var, evidence):
        parent_prob = self.prob_util(var, evidence, self.P_x_given(var, evidence))
        child_probs = 1
        for node in self.net.successors(var):
            child_probs *= self.prob_util(node, evidence, self.P_x_given(node, evidence))
        unflipped = parent_prob * child_probs

        evidence[var] = 1-evidence[var]
        parent_prob = self.prob_util(var, evidence, self.P_x_given(var, evidence))
        child_probs = 1
        for node in self.net.successors(var):
            child_probs *= self.prob_util(node, evidence, self.P_x_given(node, evidence))
        flipped = parent_prob * child_probs
        evidence[var] = 1-evidence[var]

        return unflipped/(flipped+unflipped)

    def mb(self, var):
        mb = []
        mb.extend(list(self.net.predecessors(var)))
        mb.extend(list(self.net.successors(var)))
        for node in self.net.successors(var):
            for parent in self.net.predecessors(node):
                if parent not in mb:
                    mb.append(parent)
        return mb
            

    def likelihood_weighting(self, target, N, evidence={}):
        W = {}
        for _ in range(1,N):
            x,w = self.weighted_sample(evidence)
            # print(W)
            try:
                W[x[target]] = W[x[target]] + w 
            except:
                W[x[target]] = w 
        return self.normalize(W)

    def weighted_sample(self, evidence):
        w = 1
        x = {}
        for var in list(topological_sort(self.net)):
            # print(x)
            if var in evidence:
                val = evidence[var]
                prob = self.P_x_given(var, x)
                prob = 1-prob if val == 0 else prob
                w = w * prob
            else:
                sample = random.random()
                x[var] = 1
                prob = self.P_x_given(var, x)
                x[var] = 1 if sample < prob else 0
        return x, w

    def draw(self):
        # nx.draw_graphviz(self.net, with_labels=True)
        # print(nx.get_node_attributes(self.net, 'cpt'))
        # write_dot(self.net, "test.dot")
        try:
            write_dot(self.net, "dot.dot")
        except:
            pass
        A = to_agraph(self.net)
        A.layout('dot', args='-Efontsize=10 -Efontcolor=red -Gsplines=spline')
        A.draw(outfileloc)
        # pos = graphviz_layout(self.net,prog='dot', args='')    
        # nx.draw(self.net, pos, splines='spline', with_labels=True, font_size=12, node_size=200, node_color ="white", node_border='black', edge_color="black")
        # plt.show()

def json_deep_copy(data):
    return ujson.loads(ujson.dumps(data))

class Bayes_Node():
    def __init__(self, name, parents, cpt):
        self.name = name 
        self.parents = parents
        self.cpt = cpt

def runner():
    bn = Bayes_Net()
    bn.create_from_json("bn.json")
    # bn.draw()

    starttime = time.time()
    print("Runnnig exact inference on query variable", args.query[0])
    exact_enum = bn.enumeration_ask(args.query[0])
    # print(exact_enum)
    endtime = time.time()
    exact_time = (endtime-starttime)


    print("Time\t:",exact_time,"secs\nP_False\t:",exact_enum[0],"\nP_True\t:",exact_enum[1])

    # print()
    # print('Exact enumeration time was\t\t\t', exact_time)
    # print('Likelihood weighting time was\t\t', likelihood_time)
    # print('Gibbs sampling time was\t\t\t\t\t', gibbs_time)
    # print('\nExact vs. Likelihood weighting error\t', (abs(exact_enum[1]-likelihood[1])))
    # print('Exact vs. Gibbs weighting error\t\t\t\t', (abs(exact_enum[1]-gibbs[1])))
    
if __name__ == "__main__":
    runner()



    