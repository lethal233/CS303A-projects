from IMP.IMP import Graph
from IMP.IMP import pGraph
import random
import multiprocessing as mp
import time
import math


class Worker(mp.Process):
    def __init__(self, inQ, outQ, node_num):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        self.node_num = node_num
        self.model = model
        self.graph = graph

    def run(self):

        while True:
            theta = self.inQ.get()
            while self.count < theta:
                v = random.randint(1, self.node_num)
                rr = generate_rr(v,self.model,self.graph)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num, node_num,mode):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    global worker
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), node_num))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


def sampling(epsoid, l):
    global graph, seed_size, worker
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 2
    create_worker(worker_num, n,model)
    for i in range(1, int(math.log2(n - 1)) + 1):
        s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x
        # print(theta-len(R))
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        end = time.time()
        print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k)
        print(f)
        end = time.time()
        print('node selection time', time.time() - start)
        # print(F(R, Si))
        # f = F(R,Si)
        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    # print(diff)
    _start = time.time()
    if diff > 0:
        # print('j')
        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''

    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    # print(_end - _start)
    finish_worker()
    return R


def generate_rr(v,model,graph):
    if model == 'IC':
        return generate_rr_ic(v,graph)
    elif model == 'LT':
        return generate_rr_lt(v,graph)


def node_selection(R, k):
    Sk = set()
    rr_degree = [0 for ii in range(node_num + 1)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            # print(rr_node)
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count / len(R)


'''
def node_selection(R, k):
    # use CELF to accelerate
    Sk = set()
    node_rr_set = dict()
    rr_degree = [0 for ii in range(node_num + 1)]
    matched_count = 0
    for i, rr in enumerate(R):
        for v in rr:
            if v in node_rr_set:
                node_rr_set[v].add(i)
                rr_degree[v] += 1
            else:
                node_rr_set[v] = {i}
    max_heap = list()
    for key, value in node_rr_set.items():
        max_heap.append([-len(value), key, 0])
    heapq.heapify(max_heap)
    i = 0
    covered_set = set()
    while i < k:
        val = heapq.heappop(max_heap)
        if val[2] != i:
            node_rr_set[val[1]] -= covered_set
            val[0] = -len(node_rr_set[val[1]])
            val[2] = i
            heapq.heappush(max_heap, val)
        else:
            Sk.add(val[1])
            i += 1
            covered_set |= node_rr_set[val[1]]
    return Sk, len(covered_set) / len(R)
'''


def generate_rr_ic(node,graph):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node,graph):
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        # print(candidate)
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes


def imm(epsoid, l):
    n = node_num
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l)
    Sk, z = node_selection(R, k)
    return Sk


def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def read_file(network):
    """
    read network file into a graph and read seed file into a list
    :param network: the file path of network
    """
    global node_num, edge_num, graph, seeds, pGraph
    data_lines = open(network, 'r').readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))
        pGraph.add_edge(int(start), int(end), float(weight))


if __name__ == "__main__":
    """
        define global variables:
        node_num: total number of nodes in the network
        edge_num: total number of edges in the network
        graph: represents the network
        seeds: the list of seeds
    """
    node_num = 0
    edge_num = 0
    graph = Graph()
    pGraph = pGraph()
    model = 'IC'
    """
    command line parameters
    usage: python3 IMP.py -i <graph file path> -k <the number of seeds> -m <IC or LT> -t <termination time> 
    """
    start = time.time()
    network_path = "../NetHEPT.txt"
    # network_path = "network.txt"
    model = 'IC'
    # model = 'IC'
    seed_size = 50
    # seed_size = 5
    termination = 60
    # termination = 60

    read_file(network_path)
    print(model,seed_size,termination,network_path)
    worker = []
    epsoid = 0.5
    l = 1
    seeds = imm(epsoid, l)
    # print(seeds)

    for seed in seeds:
        print(seed)

    end = time.time()
    print(end - start)
    #
    # res = ISE.calculate_influence(seeds, model, pGraph)
    #
    # print(res)
