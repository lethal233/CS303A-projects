import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
from queue import Queue
import random

core = 4
iteration = 500


class Node:
    def __init__(self, number):
        self.adjacent = []  # 记录index
        self.adjacent_weight = []
        self.number = number

    def add_adjacent_node(self, number, weight):
        self.adjacent.append(number)
        self.adjacent_weight.append(weight)


def iteration_IC_LT(N: int, seed_list: list, node_list: list, actual_end_time, model: str) -> (int, int):
    if model == 'IC':
        cul = 0
        for i in range(iteration):
            cul += one_IC_sample(seed_list, node_list)
            if time.time() > actual_end_time:
                return cul, i + 1
        return cul, iteration
    else:
        cul = 0
        for i in range(iteration):
            cul += one_LT_sample(N, seed_list, node_list)
            if time.time() > actual_end_time:
                return cul, i + 1
        return cul, iteration


def one_IC_sample(seed_list: list, node_list: list):
    q = Queue()
    activity_set = set()  # record the index !!!!!
    for sl in seed_list:
        q.put(sl)
        activity_set.add(sl)

    count = len(seed_list)
    while not q.empty():
        tmp = q.get()
        node_tmp = node_list[tmp]
        for t in range(len(node_tmp.adjacent)):
            if node_tmp.adjacent[t] not in activity_set:
                threshold = random.uniform(0,1)
                if node_tmp.adjacent_weight[t] >= threshold:
                    activity_set.add(node_tmp.adjacent[t])
                    q.put(node_tmp.adjacent[t])
                    count += 1
    return count


def one_LT_sample(N: int, seed_list: list, node_list: list):
    q = Queue()  # record the index
    activity_set = set()  # record the index
    sum_weight = []
    node_threshold = []
    for s in range(N + 1):
        sum_weight.append(0)
        tmp = random.uniform(0, 1)
        node_threshold.append(tmp)
    for sl in seed_list:
        q.put(sl)
        activity_set.add(sl)
    count = len(seed_list)
    while not q.empty():
        tmp = q.get()
        node_tmp = node_list[tmp]
        for t in range(len(node_tmp.adjacent)):
            if node_tmp.adjacent[t] not in activity_set:
                sum_weight[node_tmp.adjacent[t]] += node_tmp.adjacent_weight[t]
                if sum_weight[node_tmp.adjacent[t]] >= node_threshold[node_tmp.adjacent[t]]:
                    activity_set.add(node_tmp.adjacent[t])
                    q.put(node_tmp.adjacent[t])
                    count += 1
    return count


if __name__ == '__main__':

    start_time = time.time()
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    # parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-i', '--file_name', type=str, default='big.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=120)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit
    actual_end_time = start_time + time_limit

    # to initialize the seed nodes
    seed_list = []  # record the index
    with open(seed, 'r') as f:
        for l in f.readlines():
            seed_list.append(int(l.strip('\n')))
    N, E = 0, 0

    # to initialize all the nodes
    node_list = []  # record the object Node
    with open(file_name, 'r') as f:
        head = f.readline().strip('\n').split(' ')
        N = int(head[0])
        E = int(head[1])
        for n in range(N + 1):
            node_list.append(Node(n))
        for n in range(E):
            line = f.readline().strip('\n').split(' ')
            start = int(line[0])
            end = int(line[1])
            weight = float(line[2])
            node_list[start].add_adjacent_node(end, weight)

    pool = mp.Pool(core)
    result_tuple = []

    for i in range(core):
        result_tuple.append(pool.apply_async(iteration_IC_LT, args=(N, seed_list, node_list, actual_end_time, model)))

    pool.close()
    pool.join()

    total_iteration = 0
    total_cumulate = 0
    for r in result_tuple:
        tmp1, tmp2 = r.get()
        total_cumulate += tmp1
        total_iteration += tmp2
    # print(total_cumulate)
    # print(total_iteration)
    # print(time.time() - start_time)
    print(total_cumulate / total_iteration)
    sys.stdout.flush()
    sys.exit(0)
# 7147.9305
