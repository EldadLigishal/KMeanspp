import numpy as np
import pandas as pd
import sys
import mykmeanssp as km


def merge(file1, file2):
    to_merge1 = pd.read_csv(file1, sep=',')
    to_merge2 = pd.read_csv(file2, sep=',')
    result = pd.merge(to_merge1, to_merge2, how='inner')
    return result


def step1(i, vector, matrix):
    min_vector = np.full(len(vector), sys.maxsize)
    for j in range(i):
        val = np.subtract(vector, matrix[j])
        val = np.power(val, 2)
        if np.greater(min_vector, val):
            min_vector = val
    return min_vector


def step2(j, matrix):
    sum_matrix = np.sum(matrix)
    return np.divide(matrix[j], sum_matrix)


def execute(k, maxitr, epsilon, input_filename1, input_filename2):
    input_matrix = merge(input_filename1, input_filename2)
    length = len(input_matrix)
    if (k >= length) or (maxitr < 0) or (k < 0):
        print("Invalid Input! \n")
    np.random.seed(0)
    random_index = np.random.choice(0, length)
    centroids = np.zeros(k)
    centroids[0] = copy(input_matrix.get(random_index))
    d = np.zeros(length)
    prob = np.zeros(length)
    i = 1
    while i < k:
        for l in range(length):
            d[l] = step1(i, input_matrix[l], centroids)

        for j in range(length):
            prob[j] = step2(j, d)
            index = np.random.choice(1, length, p=prob[j])
            centroids[i] = input_matrix[index]
        i += 1
    km.fit(k,maxitr,epsilon,d,n,input_matrix,centroids);


# main
# sys.argv is the list of command-line arguments.
# len(sys.argv) is the number of command-line arguments.
# sys.argv[0] is the program i.e. the script name.
input_argv = sys.argv
input_argc = len(sys.argv)
if input_argc == 6:
    # update max_itr if needed
    # (k, maxItr, epsilon, inputFile1, inputFile2)
    execute(int(input_argv[1]), int(input_argv[2]), float(input_argv[3]), input_argv[4], input_argv[5])
else:
    # (k, epsilon, inputFile1, inputFile2)
    execute(int(input_argv[1]), 300, float(input_argv[2]), input_argv[3], input_argv[4])
