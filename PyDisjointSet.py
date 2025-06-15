"""
并查集(Disjoint Set)数据结构的实现

这个模块提供了一个高效的并查集实现，支持动态添加元素、合并集合和查找元素所属集合。
使用路径压缩和按秩合并优化以达到近乎 O(1) 的操作复杂度。
"""


class DisjointSet:
    """
    并查集类的实现

    支持动态维护不相交集合，提供元素添加、集合合并、查找和连通性判断等操作。
    使用路径压缩和按秩合并两种优化策略。

    属性:
        parent (dict[int, int]): 存储每个节点的父节点
        rank (dict[int, int]): 存储每个根节点的秩（树的近似高度）
    """

    def __init__(self, length: int = 0):
        """
        初始化并查集

        参数:
            length: 可选，初始化指定数量的元素（0到length-1）。
                   若为0则创建空并查集。
        """
        self.parent: dict[int, int] = {i: i for i in range(length)} if length else {}
        self.rank: dict[int, int] = {i: 0 for i in range(length)} if length else {}

    def find(self, x: int):
        """
        查找元素x所属集合的根节点

        使用路径压缩优化：在查找过程中将路径上的所有节点直接连接到根节点。

        参数:
            x: 要查找的元素

        返回:
            int: x所属集合的根节点
        """
        if self.parent.get(x) != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x: int, y: int):
        """
        合并包含x和y的两个集合

        使用按秩合并优化：总是将秩较小的树连接到秩较大的树上。

        参数:
            x: 第一个集合中的元素
            y: 第二个集合中的元素
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1  # 秩相同，选择一个作为根，并增加其秩

    def is_connected(self, x: int, y: int):
        """
        判断两个元素是否属于同一个集合

        参数:
            x: 第一个元素
            y: 第二个元素

        返回:
            bool: 如果x和y属于同一个集合则返回True，否则返回False
        """
        return self.find(x) == self.find(y)

    def add(self, x: int):
        """
        向并查集中添加新元素

        如果元素已存在，则不做任何操作。

        参数:
            x: 要添加的新元素
        """
        if x not in self.parent:
            self.parent[x] = x  # 初始时，每个元素自己作为根节点
            self.rank[x] = 0  # 初始秩为0


if __name__ == "__main__":
    # 示例使用
    dsu = DisjointSet()
    dsu.add(1)
    dsu.add(2)
    dsu.add(3)
    dsu.add(4)
    dsu.union(1, 2)
    dsu.union(3, 4)
    print(dsu.find(2))  # 输出：1 (因为1和2已经合并，且路径压缩后1是根)
    print(dsu.find(3))  # 输出：3 (因为3和4已经合并，且路径压缩后3是根)
