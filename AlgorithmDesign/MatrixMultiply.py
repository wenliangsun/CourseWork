#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'MatrixMutiply'
__author__ = 'swl'
__mtime__ = '11/6/18'

矩阵相乘的三种运算与Python自带的矩阵相乘作对比。
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def traditional_matrix_multiply(A, B, n):
    """
    Traditional matrix multiply
    """
    C = np.empty((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            tmp = 0
            for k in range(n):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

    return C


def divide_matrix_multiply(A, B, n):
    """分治法"""
    if n == 2:
        C11 = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
        C12 = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
        C21 = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
        C22 = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    else:
        A11 = np.mat(A[0:n // 2, 0:n // 2])
        A12 = np.mat(A[0:n // 2, n // 2:n])
        A21 = np.mat(A[n // 2:n, 0:n // 2])
        A22 = np.mat(A[n // 2:n, n // 2:n])
        B11 = np.mat(B[0:n // 2, 0:n // 2])
        B12 = np.mat(B[0:n // 2, n // 2:n])
        B21 = np.mat(B[n // 2:n, 0:n // 2])
        B22 = np.mat(B[n // 2:n, n // 2:n])

        C11 = A11 * B11 + A12 * B21
        C12 = A11 * B12 + A12 * B22
        C21 = A21 * B11 + A22 * B21
        C22 = A21 * B12 + A22 * B22

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C


def Strassen_matrix_multiply(A, B, n):
    """Strassen法"""
    if n == 2:
        M1 = A[0, 0] * (B[0, 1] - B[1, 1])
        M2 = (A[0, 0] + A[0, 1]) * B[1, 1]
        M3 = (A[1, 0] + A[1, 1]) * B[0, 0]
        M4 = A[1, 1] * (B[1, 0] - B[0, 0])
        M5 = (A[0, 0] + A[1, 1]) * (B[0, 0] + B[1, 1])
        M6 = (A[0, 1] - A[1, 1]) * (B[1, 0] + B[1, 1])
        M7 = (A[0, 0] - A[1, 0]) * (B[0, 0] + B[0, 1])
    else:
        A11 = np.mat(A[0:n // 2, 0:n // 2])
        A12 = np.mat(A[0:n // 2, n // 2:n])
        A21 = np.mat(A[n // 2:n, 0:n // 2])
        A22 = np.mat(A[n // 2:n, n // 2:n])
        B11 = np.mat(B[0:n // 2, 0:n // 2])
        B12 = np.mat(B[0:n // 2, n // 2:n])
        B21 = np.mat(B[n // 2:n, 0:n // 2])
        B22 = np.mat(B[n // 2:n, n // 2:n])

        M1 = A11 * (B12 - B22)
        M2 = (A11 + A12) * B22
        M3 = (A21 + A22) * B11
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A22) * (B11 + B22)
        M6 = (A12 - A22) * (B21 + B22)
        M7 = (A11 - A21) * (B11 + B12)

    C11 = M5 + M4 - M2 + M6
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 + M1 - M3 - M7

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C


def main():
    N = [2 ** x for x in range(2, 10)]
    C1_times = []
    C2_times = []
    C3_times = []
    C4_times = []

    for n in N:
        A = np.random.randint(0, 2, size=(n, n))
        B = np.random.randint(0, 2, size=(n, n))
        A = np.mat(A)
        B = np.mat(B)

        # C1 = np.dot(A, B)
        time_start = time.time()
        C1 = A * B
        time_end = time.time()
        C1_times.append(time_end - time_start)
        time_start = time.time()
        C2 = traditional_matrix_multiply(A, B, n)
        time_end = time.time()
        C2_times.append(time_end - time_start)
        time_start = time.time()
        C3 = divide_matrix_multiply(A, B, n)
        time_end = time.time()
        C3_times.append(time_end - time_start)
        time_start = time.time()
        C4 = Strassen_matrix_multiply(A, B, n)
        time_end = time.time()
        C4_times.append(time_end - time_start)

    plt.figure(1)
    plt.subplot(221)
    plt.semilogx(N, C1_times, 'r-*')
    plt.xlabel("Matrix Dimension(n)")
    plt.ylabel("Time(s)")
    plt.title("Python Method")
    plt.subplot(222)
    plt.semilogx(N, C2_times, 'b-*')
    plt.xlabel("Matrix Dimension(n)")
    plt.ylabel("Time(s)")
    plt.title("Direct Method")
    plt.subplot(223)
    plt.semilogx(N, C3_times, 'y-*')
    plt.xlabel("Matrix Dimension(n)")
    plt.ylabel("Time(s)")
    plt.title("Divide Method")
    plt.subplot(224)
    plt.semilogx(N, C4_times, 'g-*')
    plt.xlabel("Matrix Dimension(n)")
    plt.ylabel("Time(s)")
    plt.title("Strasses Method")

    plt.figure(2)
    plt.semilogx(N, C1_times, 'r-*', N, C2_times, 'b-*', N, C3_times, 'y-*', N, C4_times, 'g-*')
    plt.xlabel("Matrix Dimension(n)")
    plt.ylabel("Times(s)")
    plt.show()


if __name__ == '__main__':
    main()
