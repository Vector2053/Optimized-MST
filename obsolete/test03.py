# """
# import time
# from collections import deque
#
# import matplotlib
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# from scipy.spatial import cKDTree
# from scipy.spatial.distance import squareform, pdist
#
# from PyLinkCutTree import PyLinkCutTree
#
# matplotlib.use('TkAgg')
#
#
# def euclidean_distance(point1, point2):
#     return np.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(point1, point2)))
#
#
# class MSTBuilder:
#     def __init__(self):
#         self.edges = {}
#         self.nodes = []
#         self.kd_tree = None
#         self.lct = PyLinkCutTree()
#         self.reset()
#
#     def reset(self):
#         self.edges = {}
#         self.nodes = []
#         self.kd_tree = None
#         self.lct = PyLinkCutTree()
#
#     def divide_data(self):
#         # 初始化栈，压入初始任务
#         stack = deque([self.nodes])
#         while stack:
#             current_data = stack.pop()
#             # 添加当前节点到图
#             if len(current_data) <= 1:
#                 continue
#
#             points = np.array(current_data)
#
#             variances = np.var(points, axis=0)
#             split_dim = np.argmax(variances)
#
#             # 获取当前节点在划分维度上的值
#             split_value = np.median(points[:, split_dim])
#
#             # 仅在选定维度上比较
#             upper_mask = points[:, split_dim] >= split_value
#             lower_mask = points[:, split_dim] < split_value
#
#             upper_indices = np.where(upper_mask)[0]
#             lower_indices = np.where(lower_mask)[0]
#
#             # 划分区域
#             upper = [current_data[i] for i in upper_indices]
#             lower = [current_data[i] for i in lower_indices]
#             self.process_temp_sets(upper, lower, split_dim, split_value)
#             # 压栈处理子区域（注意顺序：先压 lower，再压 upper 以保证处理顺序）
#             if lower:
#                 stack.append(lower)
#
#             if upper:
#                 stack.append(upper)
#
#     def _add_edge(self, u, v, weight):
#         if u == v:
#             return
#         u, v = sorted([u, v], key=lambda x: str(x))
#         if self.lct.is_connection(u, v):
#             max_edge_lct, max_weight_lct = self.lct.get_max_edge(u, v)
#             if max_weight_lct <= weight:
#                 return
#             if max_edge_lct:
#                 self.lct.cut(max_edge_lct[0], max_edge_lct[1])
#                 self.edges.pop(max_edge_lct)
#         self.lct.link(u, v, weight=weight)
#         self.edges[(u, v)] = weight
#
#     def process_temp_sets(self, upper, lower, split_dim, split_value):
#         if not (upper and lower):
#             return
#         upper = np.array(upper)
#         lower = np.array(lower)
#         upper_bound = split_value
#         lower_bound = split_value
#         for point in upper:
#             line_node = np.array(point)
#             line_node[split_dim] = split_value
#             if self._is_gabriel_edge_single(point, line_node):
#                 if point[split_dim] > upper_bound:
#                     upper_bound = point[split_dim]
#
#         for point in lower:
#             line_node = np.array(point)
#             line_node[split_dim] = split_value
#             if self._is_gabriel_edge_single(point, line_node):
#                 if point[split_dim] < lower_bound:
#                     lower_bound = point[split_dim]
#         upper_filter = [point for point in upper if point[split_dim] <= upper_bound]
#         lower_filter = [point for point in lower if point[split_dim] >= lower_bound]
#         for p1 in upper_filter:
#             for p2 in lower_filter:
#                 if np.all(p1 == p2):
#                     continue
#                 if self._is_gabriel_edge(p1, p2):
#                     self._add_edge(tuple(p1), tuple(p2), euclidean_distance(p1, p2))
#
#     def build_mst(self, data):
#         self.reset()
#         if len(data) == 0:
#             return
#         self.nodes = np.unique(data, axis=0)
#         self.kd_tree = cKDTree(self.nodes)
#         self.divide_data()
#
#     def _is_gabriel_edge(self, u, v):
#         mid_point = (u + v) / 2
#         radius = np.linalg.norm(u - v) / 2
#
#         # # 查询圆内点（扩展半径防浮点误差）
#         candidates = self.kd_tree.query_ball_point(
#             mid_point,
#             r=radius * (1 + 1e-8),
#             p=2,
#             return_length=True
#         )
#         return candidates <= 2
#
#     def _is_gabriel_edge_single(self, u, u_projection):
#         mid_point = (u + u_projection) / 2
#         radius = np.linalg.norm(u - u_projection) / 2
#
#         # # 查询圆内点（扩展半径防浮点误差）
#         candidates = self.kd_tree.query_ball_point(
#             mid_point,
#             r=radius * (1 + 1e-8),
#             p=2,
#             return_length=True
#         )
#         return candidates == 1
#
#     def plot_mst(self):
#         graph = nx.Graph()
#         for (u, v), weight in self.edges.items():
#             graph.add_edge(u, v, weight=weight)
#         pos = {node: node for node in graph.nodes()}
#         nx.draw_networkx_nodes(graph, pos, node_color='green', node_size=50)
#         nx.draw_networkx_edges(graph, pos, width=1.5)
#         plt.title('Minimum Spanning Tree')
#         plt.show()
#
#
# def validate_mst(builder):
#     graph = nx.Graph()
#     for (u, v), weight in builder.edges.items():
#         graph.add_edge(u, v, weight=weight)
#     n = len(graph.nodes)
#     if n == 0:
#         return True  # 空图视为有效
#
#     if not nx.is_tree(graph):
#         return False
#
#     # 生成所有点对的完全图
#     # 提取坐标并计算距离矩阵
#     points = np.array([node for node in graph.nodes])
#     distance_matrix = squareform(pdist(points, 'euclidean'))
#     # 创建完全图并添加带权重的边
#     new_graph = nx.Graph()
#     edges = [(tuple(points[i]), tuple(points[j]), distance_matrix[i, j])
#              for i in range(n) for j in range(i + 1, n)]
#     new_graph.add_weighted_edges_from(edges)
#
#     # 使用Prim算法计算MST
#     mst = nx.minimum_spanning_tree(new_graph, algorithm='prim')
#
#     return np.isclose(graph.size(weight='weight'), mst.size(weight='weight'))
#
#
# # 测试用例
# if __name__ == "__main__":
#     success_count = 0
#     trials = 10
#     point_set_size = 100
#     dimension = 4
#     test_builder = MSTBuilder()
#     # np.random.seed(42)
#     start_time = time.perf_counter()
#     count_validate_time = 0
#     for _ in range(trials):
#         data = np.random.rand(point_set_size, max(dimension, 2))
#         # data = np.array([[4, 5, 7], [4, 3, 6], [4, 2, 4], [3, 2, 1], [1, 1, 0],
#         #                  [3, 3, 3], [4, 1, 5], [4, 5, 1], [2, 8, 3], [4, 6, 2],
#         #                  [5, 2, 1], [5, 5, 5], [6, 8, 4], [7, 6, 2], [8, 5, 7],
#         #                  [7, 8, 9], [9, 9, 8]])
#         test_builder.reset()
#         test_builder.build_mst(data)
#         validate_start = time.perf_counter()
#         if validate_mst(test_builder):
#             success_count += 1
#         count_validate_time += (time.perf_counter() - validate_start)
#     end_time = time.perf_counter()
#     print(f"正确率: {success_count / trials * 100:.2f}%")
#     print(f'执行时间: {end_time - start_time - count_validate_time:.4f}秒')
#     print(f'验证时间: {count_validate_time:.4f}秒')
#     if dimension <= 2:
#         test_builder.plot_mst()
# """
