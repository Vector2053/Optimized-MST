import numpy as np

from OptimizedMST import MSTBuilder


def main():
    # 生成随机测试数据
    # data = np.array([[4, 5, 7], [4, 3, 6], [4, 2, 4], [3, 2, 1], [1, 1, 0],
    #                  [3, 3, 3], [4, 1, 5], [4, 5, 1], [2, 8, 3], [4, 6, 2],
    #                  [5, 2, 1], [5, 5, 5], [6, 8, 4], [7, 6, 2], [8, 5, 7],
    #                  [7, 8, 9], [9, 9, 8]])
    np.random.seed(42)  # 设置随机种子以保证结果可复现
    n_points = 30
    data = np.random.rand(n_points, 2)  # 生成30个2D随机点

    # 创建MST构建器实例
    mst_builder = MSTBuilder()

    # 构建最小生成树
    try:
        # 使用第一个点作为初始点
        mst_builder.build_mst(data)

        # 绘制MST
        mst_builder.plot_mst()

    except Exception as e:
        print(f"构建MST时发生错误: {str(e)}")


if __name__ == "__main__":
    main()
