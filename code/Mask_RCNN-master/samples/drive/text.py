import numpy as np
import matplotlib as mlt
import PIL as Image
import os
import sys


# mask = Image.open('../../images/12283150_12d37e6389_z.jpg'
# _idx = [True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True, True, True]
# class_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# class_ids = class_ids[_idx]
# print("1", class_ids)
#
# _idx = [True, True, True, True, True, True, True, True, True, True, True, True,
#         True, True]
# class_ids = np.array([4, 7, 13, 16, 17, 20, 20, 21, 22, 22, 22, 22, 22, 29])
# class_ids = class_ids[_idx]
# print("2", class_ids)


def test(*array, arg1=None, ):
    print(arg1)
    print(array)


test(arg1=3)
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7])
c = np.array([9, 10])
print(a)
d = np.concatenate([a, b, c])
print(d, type(d))

RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
nr = np.reshape(RPN_BBOX_STD_DEV, [1, 1, 4])
print("nr", nr)
