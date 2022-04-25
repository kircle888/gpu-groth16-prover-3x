import math
import numpy as np


def pippenger(n, b, c):
    m = pow(2, c)
    w = b / c
    return {
        "名称": "Pippenger",
        "总计算量": n * w + 2 * m * w + w,
        "最长路径": n * w + 2 * m * w + w,
        "线程数": 1,
        "额外空间": m
    }


def para_straus(n, b, c, r):
    m = pow(2, c)
    w = b / c
    t = n / r
    return {
        "名称": "ParaStraus",
        "总计算量": n * w + b * t + t,
        "最长路径": r * w + b + math.log2(t),
        "线程数": t,
        "额外空间": m * n + t
    }


def para_pippenger(n, b, c):
    m = pow(2, c)
    w = b / c
    t = m * w
    return {
        "名称": "ParaPippenger",
        "总计算量": n * w + 2 * m * w + w,
        "最长路径": n / m + 2 * m + w,
        "线程数": t,
        "额外空间": t
    }


def to_Bytes(point_count):
    return point_count * 576


def to_visual(byte_count):
    if byte_count < 1024:
        return "%.1fB" % (byte_count)
    elif byte_count < 1024 * 1024:
        return "%.1fKB" % (byte_count / 1024)
    elif byte_count < 1024 * 1024 * 1024:
        return "%.1fMB" % (byte_count / 1024 / 1024)
    else:
        return "%.1fGB" % (byte_count / 1024 / 1024 / 1024)


def print_data(data):
    print(data['名称'])
    for k, v in data.items():
        if k == "额外空间":
            print(k, to_visual(to_Bytes(v)))
        else:
            print(k, v)


if __name__ == '__main__':
    print_data(pippenger(1000000, 753, 7))
    print_data(para_straus(1000000, 753, 6, 32))
    print_data(para_pippenger(1000000, 753, 10))
