# class LCTNode:
#     def __init__(self, value, is_edge=False, edge=None, weight=0):
#         self.value = value
#         self.left = None  # 左子节点
#         self.right = None  # 右子节点
#         self.parent = None  # 父节点
#         self.reverse = False  # 翻转标记（用于evert操作）
#         self.edge = edge if is_edge else None
#         self.weight = weight if is_edge else 0
#         self.max_edge = self.edge
#         self.max_weight = self.weight
#
#     def __str__(self):
#         return str(self.value)
#
#     def __eq__(self, other):
#         return self is other
#
#     def is_left(self):
#         return self.parent.left == self
#
#     def update(self):
#         self.max_weight = self.weight
#         self.max_edge = self.edge
#         # 递归比较左右子树的最大边
#         for child in [self.left, self.right]:
#             if child and child.max_weight > self.max_weight:
#                 self.max_weight = child.max_weight
#                 self.max_edge = child.max_edge
#
#     def push_down(self):
#         y = self.left
#         z = self.right
#         if self.reverse:
#             if y:
#                 y.reverse ^= True
#             if z:
#                 z.reverse ^= True
#             self.reverse ^= True
#             self.left, self.right = self.right, self.left
#
#     def is_root(self):
#         return self.parent is None or (self.parent.left != self and self.parent.right != self)
#
#     def rotate(self):
#         y = self.parent
#         z = y.parent
#         b = self.is_left()
#         if not y.is_root():
#             if z.right == y:
#                 z.right = self
#             else:
#                 z.left = self
#         self.parent = z
#         y.parent = self
#         if b:
#             if self.right:
#                 self.right.parent = y
#             y.left = self.right
#             self.right = y
#         else:
#             if self.left:
#                 self.left.parent = y
#             y.right = self.left
#             self.left = y
#         y.update()
#         self.update()
#
#     def splay(self):
#         queue = [self]
#         i = self
#         while not i.is_root():
#             queue.append(i.parent)
#             i = i.parent
#         while queue:
#             q = queue.pop()
#             q.push_down()
#         while not self.is_root():
#             y = self.parent
#             z = y.parent
#             if not y.is_root():
#                 self.rotate() if (y.left == self) ^ (z.left == y) else y.rotate()
#             self.rotate()
#
#
# def get_point_decorator(func):
#     def wrapper(self, point1, point2=None, weight=0):
#         if point2 is None and weight == 0:
#             point1 = self._get_point(point1)
#             return func(self, point1)
#         elif point2 is not None and weight == 0:
#             point1 = self._get_point(point1)
#             point2 = self._get_point(point2)
#             return func(self, point1, point2)
#         elif point2 is not None and weight != 0:
#             point1 = self._get_point(point1)
#             point2 = self._get_point(point2)
#             return func(self, point1, point2, weight)
#
#     return wrapper
#
#
# def access(x):
#     t = None
#     while x:
#         x.splay()
#         x.right = t
#         x.update()
#         t = x
#         x = x.parent
#
#
# def make_root(x):
#     access(x)
#     x.splay()
#     x.reverse ^= True
#
#
# def find_root(x):
#     access(x)
#     x.splay()
#     while x.left:
#         x = x.left
#     return x.value
#
#
# def split(x, y):
#     make_root(x)
#     access(y)
#     y.splay()
#
#
# class LinkCutTree:
#     def __init__(self):
#         self.nodes = {}
#         self.edges = {}
#
#     def add_point(self, point):
#         if point not in self.nodes:
#             self.nodes[point] = LCTNode(point)
#
#     def _get_point(self, point):
#         if point not in self.nodes:
#             self.add_point(point)
#         return self.nodes.get(point)
#
#     def _cut(self, x, y):
#         if find_root(x) != find_root(y):
#             print("两点不相连！")
#             return
#         split(x, y)
#         b = y.left if x.is_left() else y.right
#         if x.right is not None or x.parent != y or not b:
#             print("两点不直接相连！")
#             return
#         y.left = None
#         x.parent = None
#
#     def _link(self, x, y):
#         make_root(x)
#         x.parent = y
#
#     @get_point_decorator
#     def link(self, x, y, weight):
#         if find_root(x) == find_root(y):
#             print("两点已在同一个连通分量中！")
#             return
#         z = LCTNode((x.value, y.value), True, (x.value, y.value), weight)
#         self._link(x, z)
#         self._link(z, y)
#         self.edges[(x.value, y.value)] = z
#
#     @get_point_decorator
#     def cut(self, x, y):
#         z = self.edges.pop((x.value, y.value), None)
#         if z is None:
#             z = self.edges.pop((y.value, x.value), None)
#         if z is None:
#             return
#         self._cut(x, z)
#         self._cut(z, y)
#
#     @get_point_decorator
#     def is_connection(self, x, y):
#         return find_root(x) == find_root(y)
#
#     @get_point_decorator
#     def get_max_edge(self, x, y):
#         if find_root(x) != find_root(y):
#             return (None, None), None
#         split(x, y)
#         return y.max_edge, y.max_weight
#
#
# if __name__ == "__main__":
#     lct = LinkCutTree()
#     lct.link(1, 2, 3)
#     lct.link(1, 3, 5)
#     lct.cut(1, 3)
#     lct.link(2, 3, 3.0)
#     lct.cut(2, 3)
#     lct.link(1, 3, 2.4)
#     print(lct.get_max_edge(2, 3))
