import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from PyDisjointSet import DisjointSet

matplotlib.use('TkAgg')
"""
最小生成树(MST)构建器模块

这个模块提供了一个高效的最小生成树构建实现，使用基于空间分割的启发式方法来减少边的数量。
主要包含 MSTBuilder 类，用于构建和可视化最小生成树。
"""


class MSTBuilder:
    """
    最小生成树构建器类

    使用空间分割和局部连接策略构建最小生成树的实现。通过减少候选边的数量来优化性能。

    属性:
        edges (dict): 存储最终MST中的边，格式为 {(node1, node2): weight}
        candidate_edges (set): 存储候选边的集合，格式为 (node1, node2, weight)
        temp_sets (dict): 存储临时节点集
        init_tree (dict): 存储初始树结构
        curr_node_areas (dict): 存储当前节点的影响区域
        nodes (np.ndarray): 存储所有节点坐标
        kd_tree (cKDTree): 用于快速近邻查询的KD树
    """

    def __init__(self):
        self.edges: dict[tuple[int, int], float] = {}
        self.candidate_edges: set[tuple[int, int, float]] = set()
        self.temp_sets: dict[int, np.ndarray] = {}
        self.init_tree: dict[int, np.ndarray] = {}
        self.curr_node_areas: dict[int, np.ndarray] = {}
        self.nodes: np.ndarray | None = None
        self.kd_tree: cKDTree | None = None
        self.reset()

    def reset(self):
        self.edges.clear()
        self.candidate_edges.clear()
        self.temp_sets.clear()
        self.init_tree.clear()
        self.curr_node_areas.clear()
        self.nodes = None
        self.kd_tree = None

    def divide_data(self, data_idx: np.ndarray, initial_idx: np.ndarray | int, data_dimension: int) -> None:
        """
        递归地将数据空间分割成子区域，并为每个区域建立局部连接。

        参数:
            data_idx: 数据点的索引数组
            initial_idx: 初始节点的索引
            data_dimension: 数据的维度
        """
        # 初始化栈，压入初始任务
        stack = deque([(data_idx, initial_idx, -1)])
        pca = PCA(n_components=1)
        pca.fit(self.nodes)
        direction = pca.components_[0]
        while stack:
            current_idx, node_idx, parent_idx = stack.pop()
            # 添加当前节点到图
            if len(current_idx) <= 1:
                continue
            points = self.nodes[current_idx]
            node = self.nodes[node_idx]
            dists_sq = np.sum((points - node) ** 2, axis=1)  # 计算所有点到d的距离

            upper_mask = np.ones(len(points), dtype=bool)
            lower_mask = np.ones(len(points), dtype=bool)

            for dim in range(data_dimension):
                if direction[dim] >= 0:
                    # 正分量：使用大于等于划分
                    upper_mask &= (points[:, dim] >= node[dim])
                    lower_mask &= (points[:, dim] <= node[dim])
                else:
                    # 负分量：使用小于等于划分
                    upper_mask &= (points[:, dim] <= node[dim])
                    lower_mask &= (points[:, dim] >= node[dim])

            # 生成区域掩码（排除d自身）
            node_sub_idx = np.where((current_idx == node_idx))[0][0]
            # upper_mask = np.all(points >= node, axis=1)
            # lower_mask = np.all(points <= node, axis=1)
            temp_mask = ~(upper_mask | lower_mask)
            lower_mask[node_sub_idx] = False
            upper_mask[node_sub_idx] = False  # 排除d自身
            temp_mask[node_sub_idx] = False
            for mask in [lower_mask, upper_mask]:
                if np.any(mask):
                    min_idx: int | np.ndarray = current_idx[mask][np.argmin(dists_sq[mask])]
                    self._add_edge(node_idx, min_idx, np.sqrt(dists_sq[mask].min()))
                    # mid_idx = current_idx[mask][len(current_idx[mask]) // 2]
                    stack.append((current_idx[mask], min_idx, node_idx))

            if np.any(temp_mask):
                self.temp_sets[node_idx] = current_idx[temp_mask]

            self.curr_node_areas[node_idx] = np.concatenate([
                current_idx[upper_mask],
                current_idx[lower_mask],
                [node_idx],
            ])
            if parent_idx >= 0:
                self.curr_node_areas[node_idx] = np.append(self.curr_node_areas[node_idx], parent_idx)

            self.init_tree[node_idx] = np.concatenate([
                current_idx[upper_mask],
                current_idx[lower_mask],
            ])

    def _add_edge(self, i: int, j: int, weight) -> None:
        if i == j:
            return
        i, j = sorted([i, j])
        self.candidate_edges.add((i, j, weight))

    def process_temp_sets(self, initial_idx) -> None:
        """
        处理临时集合中的节点，确定最终的边连接。
        使用广度优先搜索遍历节点，并验证连接的有效性。

        参数:
            initial_idx: 初始节点的索引
        """
        visited = set()
        queue = deque([initial_idx])
        rev_bfs_sequence = deque()
        while queue:
            node_idx = queue.popleft()
            if node_idx in visited:
                continue
            visited.add(node_idx)
            rev_bfs_sequence.appendleft(node_idx)
            # 直接访问索引数组
            if node_idx in self.init_tree:
                queue.extendleft(self.init_tree[node_idx])

        for depth_idx in rev_bfs_sequence:
            temp = self.temp_sets.get(depth_idx)
            if temp is None:
                continue
            curr_nodes_idx = self.curr_node_areas.get(depth_idx, np.array([depth_idx]))
            curr_nodes = self.nodes[curr_nodes_idx]
            for p_idx in temp:
                point = self.nodes[p_idx]
                # 预处理，批量查询
                centers = (curr_nodes + point) / 2
                _, nearest_idx = self.kd_tree.query(centers, k=1)
                # 如果以node和point为直径的圆内不存在其他的点，才连接这两个点，能够节约大量时间和资源
                # node和point的中点是圆心，距离圆心最近的点不是node或point就能证明圆内部有其他点
                # 缺陷是如果有点恰好位于圆上，距离圆心最近的点可能仍会选中node或point
                nearest_point = self.nodes[nearest_idx]
                is_point = np.all(nearest_point == point, axis=1)
                is_node = np.all(nearest_point == curr_nodes, axis=1)
                valid_mask = is_point | is_node
                for i, valid in enumerate(valid_mask):
                    if valid:
                        node_idx = (curr_nodes_idx[i])
                        distance = np.linalg.norm(self.nodes[p_idx] - self.nodes[node_idx])
                        self._add_edge(p_idx, int(node_idx), distance)
                curr_nodes = np.vstack((curr_nodes, point))
                curr_nodes_idx = np.append(curr_nodes_idx, p_idx)
            del self.temp_sets[depth_idx]
            del self.curr_node_areas[depth_idx]

    def construct_mst(self) -> None:
        """
        使用Kruskal算法构建最终的最小生成树。
        从所有候选边中选择权重最小的边，同时避免形成环。
        """
        all_edges: list[tuple[int, int, float]] = list(self.candidate_edges)
        all_edges.sort(key=lambda x: x[-1])
        total_nodes: int = len(self.nodes)
        dsu = DisjointSet(total_nodes)
        for i, j, w in all_edges:
            if total_nodes - 1 == len(self.edges):
                break
            if dsu.is_connected(i, j):
                continue
            dsu.union(i, j)
            self.edges[(i, j)] = w

    def build_mst(self, data: np.ndarray) -> None:
        """
        构建输入数据的最小生成树。

        参数:
            data: 形状为 (n, d) 的numpy数组，其中n是点的数量，d是维度
        """
        self.reset()
        if len(data) == 0:
            return
        self.nodes = np.unique(data, axis=0)
        data_idx = np.arange(len(self.nodes))
        self.kd_tree = cKDTree(self.nodes)
        _, mid_index = self.kd_tree.query(self.nodes.mean(axis=0), k=1)
        initial_idx = data_idx[mid_index]
        self.divide_data(data_idx, initial_idx, len(data[0]))
        self.process_temp_sets(initial_idx)
        self.construct_mst()

    def plot_mst(self) -> None:
        """
        使用matplotlib和networkx可视化最小生成树。
        注意：仅适用于2维数据。
        """

        if self.nodes is None or self.nodes.shape[1] != 2:
            raise ValueError("只能绘制二维数据的MST")
        graph: nx.Graph = nx.Graph()
        for (i, j), weight in self.edges.items():
            graph.add_edge(tuple(self.nodes[i]), tuple(self.nodes[j]), weight=weight)
        pos = {node: node for node in graph.nodes()}
        nx.draw_networkx_nodes(graph, pos, node_color='green', node_size=50)
        nx.draw_networkx_edges(graph, pos, width=1.5)
        plt.title('Minimum Spanning Tree')
        plt.show()


def validate_mst(builder: MSTBuilder) -> bool:
    """
    验证构建的最小生成树是否正确。

    通过比较与scipy实现的结果来验证MST的正确性。

    参数:
        builder: MSTBuilder实例

    返回:
        bool: 如果MST正确则返回True，否则返回False
    """
    total_weight: int = 0
    for _, weight in builder.edges.items():
        total_weight += weight
    n = len(builder.nodes)
    if n == 0:
        return True  # 空图视为有效

    if len(builder.edges) != n - 1:
        return False

    # 生成所有点对的完全图
    # 提取坐标并计算距离矩阵
    points = np.array(builder.nodes)
    # 创建距离矩阵的 CSR 格式稀疏表示
    i, j = np.triu_indices(n, k=1)  # 只计算上三角部分
    dists = np.sqrt(((points[i] - points[j]) ** 2).sum(axis=1))
    # 创建稀疏距离矩阵
    dist_matrix = csr_matrix((dists, (i, j)), shape=(n, n))
    # 直接计算最小生成树
    mst = minimum_spanning_tree(dist_matrix.tocsr())
    # 计算总权重
    mst_weight = mst.sum()

    return np.isclose(total_weight, mst_weight)


# 测试用例
if __name__ == "__main__":
    success_count: int = 0
    trials: int = 10
    point_set_size: int = 100
    dimension: int = 4
    test_builder = MSTBuilder()
    start_time: float = time.perf_counter()
    count_validate_time: float = 0
    for _ in range(trials):
        data = np.random.rand(point_set_size, max(dimension, 2))
        test_builder.reset()
        test_builder.build_mst(data)
        validate_start: float = time.perf_counter()
        if validate_mst(test_builder):
            success_count += 1
        count_validate_time += (time.perf_counter() - validate_start)
    end_time = time.perf_counter()
    print(f"正确率: {success_count / trials * 100:.2f}%")
    print(f'执行时间: {end_time - start_time - count_validate_time:.4f}秒')
    print(f'验证时间: {count_validate_time:.4f}秒')
    if dimension <= 2:
        test_builder.plot_mst()

