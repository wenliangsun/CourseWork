#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'PolynomialSolution'
__author__ = 'swl'
__mtime__ = '11/6/18'

四种方法实现多项式求值，并使用不同的规模进行对比。
# TODO 第四种方法是什么？
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def direct_method(A, x):
    """
    T(n) = O(n^2)
    """
    n = A.shape[0]
    result = A[0]
    for i in range(1, n):
        result += A[i] * x ** i

    return result


def horner_method(A, x):
    """
    T(n) = O(n)
    """
    n = A.shape[0]
    result = A[0]
    Q = 1
    for i in range(1, n):
        Q = Q * x
        result += A[i] * Q
    return result


def third_method(A, x):
    """
    T(n) = O(n)
    """
    n = A.shape[0]
    result = A[-1]
    for i in range(1, n):
        result = result * x + A[n - i - 1]
    return result


def main():
    N = [10, 50, 100, 150, 200, 300, 400, 500, 10000, 20000, 50000, 100000]
    # N = [10, 50, 100, 150, 200, 300, 400, 500, 1000, 10000]
    # N = [10, 50, 100, 150, 200, 300, 400]
    M1_times = []
    M2_times = []
    M3_times = []
    M4_times = []

    for n in N:
        A = np.random.rand(n + 1)
        x = 0.5

        time_start = time.time()
        result1 = direct_method(A, x)
        time_end = time.time()
        M1_times.append(time_end - time_start)
        time_start = time.time()
        result2 = horner_method(A, x)
        time_end = time.time()
        M2_times.append(time_end - time_start)
        time_start = time.time()
        result3 = third_method(A, x)
        time_end = time.time()
        M3_times.append(time_end - time_start)

    plt.figure(1)
    plt.subplot(221)
    plt.semilogx(N, M1_times, 'r-*')
    plt.xlabel("N")
    plt.ylabel("Times(s)")
    plt.title("Direct Method")
    plt.subplot(222)
    plt.semilogx(N, M2_times, 'y-*')
    plt.xlabel("N")
    plt.ylabel("Times(s)")
    plt.title("Hornor Method")
    plt.subplot(223)
    plt.semilogx(N, M3_times, 'b-*')
    plt.xlabel("N")
    plt.ylabel("Times(s)")
    plt.title("Third Method")

    plt.figure(2)
    plt.semilogx(N, M1_times, 'r-*', N, M2_times, 'b-*', N, M3_times, 'y-*')
    plt.xlabel("N")
    plt.ylabel("Times(s)")

    plt.show()


if __name__ == '__main__':
    main()
