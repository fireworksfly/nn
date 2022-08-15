import numpy as np

if __name__ == '__main__':
    # 将问题抽象为张量
    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    # 初始化权重
    # 给个随机种子， 使权重的起点都是相同的
    np.random.seed(1)
    weights = 2 * np.random.random((3, 1)) - 1
    print(weights)

    # 循环，算是迭代了吧，这里比较简单
    for i in range(10000):
        # 矩阵相乘， 记住矩阵也可以叉乘哦
        z = np.dot(X, weights)
        # 使用sigmoid函数计算最终的output, 记住这里如果是个矩阵，那么矩阵里的每个数都会放进去计算
        output = 1 / (np.exp(-z) + 1)

        # 定义误差函数
        lost = y - output

        # 计算斜率
        slope = output * (1 - output)

        # 计算增量
        delta = lost * slope

        # 更新权重
        weights = weights + np.dot(X.T, delta)
    print(weights)