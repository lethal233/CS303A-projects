import multiprocessing as mp
import time
import sys
import argparse
import random
import math

core = 2
ic_time = 0

class Node:
    def __init__(self):
        self.neighbor = {}

    def add_neighbor(self, number, weight):
        self.neighbor[number] = weight


def node_selection(R: list):
    print("Rlength: %d" % (len(R)))
    S_astar = set()
    count = 0
    RR_node_set_deg = [0 for _ in range(N + 1)]  # 初始化节点的RRS的数量
    RR_node_set_index = {}  # 记录n^th节点所对应的RR集合的index

    for g in range(len(R)):
        rr = R[g]
        for gs in rr:
            RR_node_set_deg[gs] += 1
            if not RR_node_set_index.__contains__(gs):
                RR_node_set_index[gs] = []
            RR_node_set_index[gs].append(g)
    for i in range(k):
        vj = RR_node_set_deg.index(max(RR_node_set_deg))
        try:
            count += len(RR_node_set_index[vj])
        except KeyError:
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
    return S_astar, count / len(R)


def IMM(epsilon, l):
    l = l * (1 + math.log(2) / math.log(N))
    R = sampling(epsilon, l)
    Sk, _ = node_selection(R)
    return Sk


def log_C_nk():
    result = 0
    for i in range(N - k + 1, N + 1):
        result += math.log(i)
    for j in range(1, k + 1):
        result -= math.log(j)
    return result


def sampling(epsilon, l):
    R = []
    LB = 1
    epsilon_prime = math.sqrt(2) * epsilon
    lambda_prime = ((2 + 2 * epsilon_prime / 3) * (
            lcnk + l * math.log(N) + math.log(math.log2(N))) * N) / pow(
        epsilon_prime, 2)
    for i in range(1, math.floor(math.log2(N))):
        print("循环")
        x = N / pow(2, i)
        theta_i = lambda_prime / x
        print("theta_i is",theta_i)
        # st = time.time()
        while len(R) < theta_i:
            v = random.randint(1, N)
            if model == 'IC':
                rr = get_RR_IC([v])
            else:
                rr = get_RR_LT([v])
            R.append(rr)

        # print("time for loops", mid - st)

        # mid = time.time()
        # print(mid-st)
        S_i, fr = node_selection(R)
        # print("time for selection", time.time() - mid)
        if N * fr > (1 + epsilon_prime) * x:
            LB = N * fr / (1 + epsilon_prime)
            break
        # if (mid - st) * 15 + mid > end_time:
        #     break
    alpha = math.sqrt(l * math.log(N) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (lcnk + l * math.log(N) + math.log(2)))
    lambda_star = 2 * N * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsilon, -2)
    theta = lambda_star / LB
    while len(R) < theta:
        v = random.randint(1, N)
        if model == 'IC':
            rr = get_RR_IC([v])
        else:
            rr = get_RR_LT([v])
        R.append(rr)
    return R


def get_RR_mul(R, theta):
    pool = mp.Pool(core)
    result = []
    partition = math.ceil((theta - len(R)) / core)
    for i in range(core):
        result.append(pool.apply_async(get_RR, args=(partition, end_time)))
    pool.close()
    pool.join()
    for r in result:
        R += r.get()


def get_RR(partition, end_time):
    rr_tmp = []
    count = 0
    if model == 'IC':
        while count < partition:
            v = random.randint(1, N)
            rr_tmp.append(get_RR_IC([v]))
            count += 1
    else:
        while count < partition:
            v = random.randint(1, N)
            rr_tmp.append(get_RR_IC([v]))
            count += 1
    return rr_tmp


def get_RR_IC(node_set):
    global ic_time
    st1 = time.time()
    activity_set = []
    RR = []
    for i in node_set:
        RR.append(i)
        activity_set.append(i)
    while activity_set:
        new_activity_set = []
        for seed in activity_set:
            for n, w in graph[seed].neighbor.items():
                if n not in RR and random.uniform(0, 1) < w:
                    RR.append(n)
                    new_activity_set.append(n)
        activity_set = new_activity_set
    ic_time += time.time()-st1
    return RR


def get_RR_LT(node_set):
    activity_set = []
    RR = []
    for i in node_set:
        RR.append(i)
        activity_set.append(i)
    while activity_set:
        new_activity_set = []
        for seed in activity_set:
            if graph[seed].neighbor:
                idx = random.sample(graph[seed].neighbor.keys(), 1)[0]
                if idx not in RR:
                    RR.append(idx)
                    new_activity_set.append(idx)
        activity_set = new_activity_set
    return RR


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
    parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-k', '--pre_size', type=int, default=5)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.pre_size
    model = args.model
    time_limit = args.time_limit
    end_time = start_time + time_limit - 12

    graph = []
    N, E = 0, 0
    read_graph()

    ####
    lcnk = log_C_nk()
    ####
    epsilon = 0.08
    l = 1
    seed_sss = IMM(epsilon, l)
    for sss in seed_sss:
        print(sss)
    # print(time.time() - stat)
    print("ic total time is ",ic_time)
    sys.stdout.flush()

"""
每一次node_selection R 都会翻倍，时间大概是4倍的时间
10w的节点可以生成 $theta = 300w$ 的R
ic total time 178 s
node selection 时间 600 s
"""
