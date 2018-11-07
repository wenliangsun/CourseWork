#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'OneDimensionalSearch'
__author__ = 'swl'
__mtime__ = '11/7/18'
一维搜索算法
function :3*x**3 -4*x +2  ,[0,2] ,eps=0.2
"""


def fn(x):
    """目标函数"""
    return 3 * x ** 3 - 4 * x + 2


def gradient(x):
    """目标函数的梯度"""
    return 9 * x ** 2 - 4


def hessen(x):
    """目标函数的hessen矩阵(二阶导)"""
    return 18 * x


def success_failure_method(x, h, eps=0.01):
    """成功-失败法"""
    y1 = fn(x)
    while abs(h) > eps:
        y2 = fn(x + h)
        if y2 < y1:
            x = x + h
            y1 = y2
            h = 2 * h
        else:
            h = -h / 4
    return x


def zero_dot_618_method(dist, eps=0.01):
    """0.618法"""
    a, b = dist
    x1 = a + (1 - 0.618) * (b - a)
    x2 = a + 0.618 * (b - a)
    while b - a > 0:
        if b - a < eps:
            return (a + b) / 2
        elif fn(x1) > fn(x2):
            a = x1
            x1 = x2
            x2 = a + 0.618 * (b - a)
        elif fn(x1) <= fn(x2):
            b = x2
            x2 = x1
            x1 = a + (1 - 0.618) * (b - a)


def Newton_method(x, eps=0.01):
    """牛顿法"""
    while abs(gradient(x)) > eps:
        x = x - gradient(x) / hessen(x)
    return x


def bisection_method(dist, eps=0.01):
    """对分法"""
    a, b = dist
    assert gradient(a) < 0, "区间不符合要求"
    assert gradient(b) > 0, "区间不符合要求"

    while b - a > 0:
        c = (a + b) / 2
        grad_c = gradient(c)
        if grad_c == 0:
            return c
        elif b - a <= eps:
            return c
        elif grad_c <= 0:
            a = c
        elif grad_c > 0:
            b = c


if __name__ == '__main__':
    x = 0.9
    h = 2  # 初始化步长
    dist = [0, 2]
    eps = 0.01
    print("---------------------**成功-失败法**-----------------------------------\n")
    print("最优解为：", success_failure_method(x, h, eps))
    print("最小值为：", fn(success_failure_method(x, h, eps)), '\n')
    print("----------------------**0.618法**-----------------------------------\n")
    print("最优解为：", zero_dot_618_method(dist, eps))
    print("最小值为：", fn(zero_dot_618_method(dist, eps)), '\n')
    print("---------------------**牛顿法**-----------------------------------\n")
    print("最优解为：", Newton_method(x, eps))
    print("最小值为：", fn(Newton_method(x, eps)), '\n')
    print("---------------------**对分法**-----------------------------------\n")
    print("最优解为：", bisection_method(dist, eps))
    print("最小值为：", fn(bisection_method(dist, eps)))
