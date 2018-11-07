#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'UnconstrainedOptimization'
__author__ = 'swl'
__mtime__ = '11/7/18'
无约束优化问题
function = x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 4 * x1
"""
import numpy as np


def fn(X):
    """目标函数"""
    x1, x2 = X[0, 0], X[1, 0]
    return x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 4 * x1


def gradient(X):
    """目标函数的梯度"""
    x1, x2 = X[0, 0], X[1, 0]
    return np.mat([2 * x1 - 2 * x2 - 4, 4 * x2 - 2 * x1]).T


def hessen(X):
    """目标函数的hessen矩阵"""
    return np.mat([[2, -2], [-2, 4]])


def two_norm(X):
    """二范数"""
    return np.sqrt(X[0, 0] ** 2 + X[1, 0] ** 2)


def gradient_descent_method(X, eps=0.001):
    """最速梯度下降法"""
    D = - gradient(X)
    while two_norm(D) > eps:

        def fn_lmd(lmd):
            """构建的关于lambda的函数，对lambda进行使用0.618法进行一维搜索"""
            new_X = X + lmd * D
            return fn(new_X)

        def zero_dot_618_method(dist, eps=0.01):
            """0.618法"""
            a, b = dist
            x1 = a + (1 - 0.618) * (b - a)
            x2 = a + 0.618 * (b - a)
            while b - a > 0:
                if b - a < eps:
                    return (a + b) / 2
                elif fn_lmd(x1) > fn_lmd(x2):
                    a = x1
                    x1 = x2
                    x2 = a + 0.618 * (b - a)
                elif fn_lmd(x1) <= fn_lmd(x2):
                    b = x2
                    x2 = x1
                    x1 = a + (1 - 0.618) * (b - a)

        # 使用0.618法搜索最优步长
        lmd = zero_dot_618_method([-10, 10], eps=0.001)
        X = X + lmd * D
        D = gradient(X)
    return X


def Newton_method(X, eps=0.001):
    """牛顿法"""
    G = gradient(X)
    while two_norm(G) > eps:
        H = hessen(X)
        D = -H.I * G
        X += D
        G = gradient(X)
    return X


def new_Newton_method(X, eps=0.001):
    """阻尼牛顿法"""
    G = gradient(X)
    while two_norm(G) > eps:
        H = hessen(X)
        D = -H.I * G

        def fn_lmd(lmd):
            """构建的关于lambda的函数，对lambda进行使用0.618法进行一维搜索"""
            new_X = X + lmd * D
            return fn(new_X)

        def zero_dot_618_method(dist, eps=0.01):
            """0.618法"""
            a, b = dist
            x1 = a + (1 - 0.618) * (b - a)
            x2 = a + 0.618 * (b - a)
            while b - a > 0:
                if b - a < eps:
                    return (a + b) / 2
                elif fn_lmd(x1) > fn_lmd(x2):
                    a = x1
                    x1 = x2
                    x2 = a + 0.618 * (b - a)
                elif fn_lmd(x1) <= fn_lmd(x2):
                    b = x2
                    x2 = x1
                    x1 = a + (1 - 0.618) * (b - a)

        lmd = zero_dot_618_method([-10, 10], eps=0.001)
        X += lmd * D
        G = gradient(X)

    return X


def conjugate_gradient_method(X, eps=0.001):
    """共轭梯度法"""
    n = X.shape[0]
    G = gradient(X)
    for i in range(n):  # 至多n次迭代就可以找到最优解
        D = -G

        def fn_lmd(lmd):
            """构建的关于lambda的函数，对lambda进行使用0.618法进行一维搜索"""
            new_X = X + lmd * D
            return fn(new_X)

        def zero_dot_618_method(dist, eps=0.01):
            """0.618法"""
            a, b = dist
            x1 = a + (1 - 0.618) * (b - a)
            x2 = a + 0.618 * (b - a)
            while b - a > 0:
                if b - a < eps:
                    return (a + b) / 2
                elif fn_lmd(x1) > fn_lmd(x2):
                    a = x1
                    x1 = x2
                    x2 = a + 0.618 * (b - a)
                elif fn_lmd(x1) <= fn_lmd(x2):
                    b = x2
                    x2 = x1
                    x1 = a + (1 - 0.618) * (b - a)

        lmd = zero_dot_618_method([-10, 10], eps=0.001)  # 利用0.618法求解最优步长

        X += lmd * D
        G_old = G
        G = gradient(X)
        if two_norm(G) < eps:
            return X
        beta = two_norm(G) ** 2 / two_norm(G_old) ** 2
        D = -G + beta * D
    return X


def main():
    X = np.mat([1., 1.]).T  # 初始点
    eps = 0.001
    print("-----------------------**最速下降法**-----------------------------\n")
    print("最优解为：", gradient_descent_method(X, eps))
    print("最小值为：", fn(gradient_descent_method(X, eps)), '\n')
    print("-----------------------**牛顿法**-----------------------------\n")
    print("最优解为：", Newton_method(X, eps))
    print("最小值为：", fn(Newton_method(X, eps)), '\n')
    print("-----------------------**阻尼牛顿法**-----------------------------\n")
    print("最优解为：", new_Newton_method(X, eps))
    print("最小值为：", fn(new_Newton_method(X, eps)))
    print("-----------------------**共轭梯度法**-----------------------------\n")
    print("最优解为：", conjugate_gradient_method(X, eps))
    print("最小值为：", fn(conjugate_gradient_method(X, eps)))


if __name__ == '__main__':
    main()
