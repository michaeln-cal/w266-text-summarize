import numpy as np

input_list = ['all', 'this']


def check_duplicate(input_list, n):
    n_grams = [item for item in zip(*[input_list[i:] for i in range(n)])]
    dup = 0
    for i in range(len(n_grams) - 1):
        for j in range(i + 1, len(n_grams)):
            if n_grams[j] == n_grams[i]: dup = +1

    return dup


print(check_duplicate(input_list, 3))
