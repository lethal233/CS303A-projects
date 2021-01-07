# def get_RR_IC(node_set):
#     RR = set()  # index
#     activity_set = set()
#     for ns in node_set:
#         RR.add(ns)
#         activity_set.add(ns)
#     while activity_set:
#         new_activity_set = set()
#         for seed in activity_set:
#             for i in range(len(node_list_reverse[seed].adjacent)):
#                 threshold = random.uniform(0, 1)
#                 neighbor = node_list_reverse[seed].adjacent[i]
#                 if neighbor not in RR and neighbor not in new_activity_set and node_list_reverse[seed].adjacent_weight[
#                     i] >= threshold:
#                     new_activity_set.add(neighbor)
#         activity_set = new_activity_set
#         RR = RR.union(new_activity_set)
#     return RR


"""
# for key in RR_node_set_index:
#     for dei in delete_index:
#         if dei in RR_node_set_index[key]:
#             RR_node_set_index[key].remove(dei)
#             RR_node_set_deg[key] -= 1
# for key, value in RR_node_set_index.items():
#     for t in index_all:
#         if t in value:
#             value.remove(t)
#             RR_node_set_deg[key] -= 1
"""



# def node_selection(R):
#     Sk = set()
#     rr_degree = [0 for ii in range(N + 1)]
#     node_rr_set = dict()
#     # node_rr_set_copy = dict()
#     matched_count = 0
#     for j in range(0, len(R)):
#         rr = R[j]
#         for rr_node in rr:
#             # print(rr_node)
#             rr_degree[rr_node] += 1
#             if rr_node not in node_rr_set:
#                 node_rr_set[rr_node] = list()
#                 # node_rr_set_copy[rr_node] = list()
#             node_rr_set[rr_node].append(j)
#             # node_rr_set_copy[rr_node].append(j)
#     for i in range(k):
#         max_point = rr_degree.index(max(rr_degree))
#         Sk.add(max_point)
#         matched_count += len(node_rr_set[max_point])
#         index_set = []
#         for node_rr in node_rr_set[max_point]:
#             index_set.append(node_rr)
#         for jj in index_set:
#             rr = R[jj]
#             for rr_node in rr:
#                 rr_degree[rr_node] -= 1
#                 node_rr_set[rr_node].remove(jj)
#     return Sk, matched_count / len(R)