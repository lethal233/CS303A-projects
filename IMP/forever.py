import multiprocessing as mp
import time
import sys
import argparse
import random
import math

core = 4


class Node:
    def __init__(self):
        self.neighbor = {}

    def add_neighbor(self, number, weight):
        self.neighbor[number] = weight


def node_selection(R: list):
    # print("Rlength: %d" % (len(R)))
    # s = time.time()
    S_astar = set()
    count = 0
    RR_node_set_deg = [0 for _ in range(N + 1)]  # 初始化节点的RRS的数量
    RR_node_set_index = {}  # 记录n^th节点所对应的RR集合的index

    for g in range(len(R)):
        rr = R[g]
        for gs in rr:
            RR_node_set_deg[gs] += 1
            if not RR_node_set_index.__contains__(gs):
                RR_node_set_index[gs] = set()
            RR_node_set_index[gs].add(g)

    for i in range(k):
        vj = RR_node_set_deg.index(max(RR_node_set_deg))
        if vj != 0:
            count += len(RR_node_set_index[vj])
        else:
            contd = i
            for j in range(contd, k):
                for w in range(1, N + 1):
                    if w not in S_astar:
                        S_astar.add(w)
                        break
            break
        S_astar.add(vj)
        # 要把包含vj的set的index全给去掉（在反向集合字典中）
        delete_index = RR_node_set_index[vj].copy()
        for idx in delete_index:
            rr_idx = R[idx]
            for r in rr_idx:
                RR_node_set_deg[r] -= 1
                RR_node_set_index[r].remove(idx)
    # print("node selection time:", time.time() - s)
    return S_astar, count / len(R)


def IMM(epsilon, l, end_time, model, graph, N):
    l = l * (1 + math.log(2) / math.log(N))
    R = sampling(epsilon, l, end_time, model, graph, N)
    Sk, _ = node_selection(R)
    return Sk


def log_C_nk():
    result = 0
    for i in range(N - k + 1, N + 1):
        result += math.log(i)
    for j in range(1, k + 1):
        result -= math.log(j)
    return result


def sampling(epsilon, l, end_time, model, graph, N):
    R = []
    LB = 1
    epsilon_prime = math.sqrt(2) * epsilon
    lambda_prime = ((2 + 2 * epsilon_prime / 3) * (
            lcnk + l * math.log(N) + math.log(math.log2(N))) * N) / pow(
        epsilon_prime, 2)
    # a = time.time()
    get_RR_mul(R, 4000000, end_time, model, graph, N)
    # print("get RR time : ",time.time()-a)
    return R
    #
    #
    # for i in range(1, math.floor(math.log2(N))):
    #     x = N / pow(2, i)
    #
    #     theta_i = lambda_prime / x
    #     # st = time.time()
    #     print("theta_i", theta_i)
    #     a = time.time()
    #     get_RR_mul(R, theta_i, end_time, model, graph, N)
    #     print("get RR time:", time.time() - a)
    #     # while len(R) < theta_i:
    #     #     v = random.randint(1, N)
    #     #     if model == 'IC':
    #     #         rr = get_RR_IC([v])
    #     #     else:
    #     #         rr = get_RR_LT([v])
    #     #     R.append(rr)
    #
    #     # print("time for loops", mid - st)
    #
    #     # mid = time.time()
    #     # print(mid-st)
    #     S_i, fr = node_selection(R)
    #     # print("time for selection", time.time() - mid)
    #     if N * fr > (1 + epsilon_prime) * x:
    #         LB = N * fr / (1 + epsilon_prime)
    #         break
    #     # if (mid - st) * 15 + mid > end_time:
    #     #     break
    # # print("R's length is ", len(R))
    # alpha = math.sqrt(l * math.log(N) + math.log(2))
    # beta = math.sqrt((1 - 1 / math.e) * (lcnk + l * math.log(N) + math.log(2)))
    # lambda_star = 2 * N * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsilon, -2)
    # theta = lambda_star / LB
    # print("theta:", theta)
    # d = theta - len(R)
    # if d > 0:
    #     b = time.time()
    #     get_RR_mul(R, theta, end_time, model, graph, N)
    #     print("get RR time:", time.time() - b)
    # # while len(R) < theta:
    # #     v = random.randint(1, N)
    # #     if model == 'IC':
    # #         rr = get_RR_IC([v])
    # #     else:
    # #         rr = get_RR_LT([v])
    # #     R.append(rr)
    # return R


def get_RR_mul(R, theta, end_time, model, graph, N):
    pool = mp.Pool(core)
    result = []
    partition = math.ceil((theta - len(R)) / core)
    for i in range(core):
        result.append(pool.apply_async(get_RR, args=(partition, end_time, model, graph, N)))
    pool.close()
    pool.join()
    for r in result:
        R += r.get()


def get_RR(partition, end_time, model, graph, N):
    rr_tmp = []
    count = 0
    mem_limit = 0.3 * 1024 * 1024 * 1024
    mem = 0
    if model == 'IC':
        while count < partition:
            v = random.randint(1, N)
            rr = get_RR_IC([v], graph)
            mem += sys.getsizeof(rr)
            if mem >= mem_limit or time.time() >= end_time:
                # if mem>=mem_limit:
                #     print("mle")
                # else:
                #     print("tle")
                break
            rr_tmp.append(rr)
            count += 1
    else:
        while count < partition:
            v = random.randint(1, N)
            rr = get_RR_LT([v], graph)
            mem += sys.getsizeof(rr)
            if mem >= mem_limit or time.time() >= end_time:
                # if mem>=mem_limit:
                #     print("mle")
                # else:
                #     print("tle")
                break
            rr_tmp.append(rr)
            count += 1
    return rr_tmp


def get_RR_IC(node_set, graph):
    activity_set = []
    RR = set()
    for i in node_set:
        RR.add(i)
        activity_set.append(i)
    while not len(activity_set) == 0:
        new_activity_set = []
        for seed in activity_set:
            for n, w in graph[seed].neighbor.items():
                if n not in RR and random.random() < w:
                    RR.add(n)
                    new_activity_set.append(n)
        activity_set = new_activity_set
    return list(RR)


def get_RR_LT(node_set, graph):
    activity_set = []
    RR = set()
    for i in node_set:
        RR.add(i)
        activity_set.append(i)
    while not len(activity_set) == 0:
        new_activity_set = []
        for seed in activity_set:
            if graph[seed].neighbor:
                idx = random.sample(graph[seed].neighbor.keys(), 1)[0]
                if idx not in RR:
                    RR.add(idx)
                    new_activity_set.append(idx)
        activity_set = new_activity_set
    return list(RR)


def read_graph():
    global N, E, graph
    with open(file_name, 'r') as f:
        head = f.readline().strip('\n').split(' ')
        N = int(head[0])
        E = int(head[1])
        for n in range(N + 1):
            graph.append(Node())
        for n in range(E):
            line = f.readline().strip('\n').split(' ')
            start = int(line[0])
            end = int(line[1])
            weight = float(line[2])
            graph[end].add_neighbor(start, weight)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='big.txt')
    parser.add_argument('-k', '--pre_size', type=int, default=50)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.pre_size
    model = args.model
    time_limit = args.time_limit

    if k <= 50:
        end_time = start_time + time_limit * 0.34
    else:
        end_time = start_time + time_limit * 0.26

    graph = []
    N, E = 0, 0
    read_graph()

    ####
    lcnk = log_C_nk()

    ####
    epsilon = 0.08
    l = 1
    seed_sss = IMM(epsilon, l, end_time, model, graph, N)
    for sss in seed_sss:
        print(sss)
    print(time.time() - start_time)

    sys.stdout.flush()
    sys.exit(0)

"""
1w5 节点 3w2 边 50 seed 120w theta
1w5 节点 3w2 边 5 seed 105w theta
62 159 5 seed 2w theta

71.6w -> 143.3w -> 143.2w

1w5 3w2 边 500 nodes
71.6w -> 143w -> 190w
1w5 3w2 边 50 nodes
11w -> 11.43 -> 44.8w -> 89w -> 115w
62 150 5 nodes
7k->1w4->1w8
10w 25w 100 nodes 
26w -> 52w -> 104w -> 208w -> 280w
"""
