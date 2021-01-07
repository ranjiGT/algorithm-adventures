import numpy as np
import csv
from numpy import genfromtxt
import argparse


def var(numbers, mean_num):
    avg = mean_num
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return variance

def data_summary(col, labels):
    
    class_a = []
    class_b = []
    data_a = []
    data_b = []

    for i in range(len(labels)):
        if labels[i]=='A':
            class_a.append(labels[i])
            data_a.append(col[i])
        elif labels[i]=='B':
            class_b.append(labels[i])
            data_b.append(col[i])

    return data_a, data_b, class_a, class_b


def calculate_probability(x, mean, var):
    exponent = np.exp(-((x-mean)**2 / (2 * var )))
    return (1 / (np.sqrt(2 * np.pi*var))) * exponent


def Naive_Bayes(col1, col2, labels, M):

    data_a_1, data_b_1, class_a_1, class_b_1 = data_summary(col1, labels)
    data_a_2, data_b_2, class_a_2, class_b_2 = data_summary(col2, labels)


    prob_a_1 = len(class_a_1)/len(labels)
    prob_b_1 = len(class_b_1)/len(labels)

    prob_a_2 = len(class_a_2)/len(labels)
    prob_b_2 = len(class_b_2)/len(labels)

    mu_a_1 = np.mean(data_a_1)
    mu_b_1 = np.mean(data_b_1)

    mu_a_2 = np.mean(data_a_2)
    mu_b_2 = np.mean(data_b_2)

    var_a_1 = var(data_a_1, mu_a_1)
    var_b_1 = var(data_b_1, mu_b_1)

    var_a_2 = var(data_a_2, mu_a_2)
    var_b_2 = var(data_b_2, mu_b_2)


    count = 0
    for i in range(len(M)):

        gaus_a_1 = calculate_probability(col1[i], mu_a_1, var_a_1)
        gaus_b_1 = calculate_probability(col1[i], mu_b_1, var_b_1)

        gaus_a_2 = calculate_probability(col2[i], mu_a_2, var_a_2)
        gaus_b_2 = calculate_probability(col2[i], mu_b_2, var_b_2)



        p_a = prob_a_1*gaus_a_1*gaus_a_2
        p_b = prob_b_1*gaus_b_1*gaus_b_2


        if p_a > p_b:
            clss = "A"
        elif p_b > p_a:
            clss = "B"
        else:
            pass

        if labels[i]!=clss:
            count+=1


    print(mu_a_1, end=" ")
    print(var_a_1, end=" ")
    print(mu_a_2, end=" ")
    print(var_a_2, end=" ")
    print(prob_a_1)
    print(mu_b_1, end=" ")
    print(var_b_1, end=" ")
    print(mu_b_2, end=" ")
    print(var_b_2, end=" ")
    print(prob_b_1)
    print(count)


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                     help='file for training')

    args = parser.parse_args()
    file_path = args.data

    x = genfromtxt(file_path, delimiter=',',dtype=str)
    M = np.array(x)

    labels = M[:,0]
    col1 = M[:, 1].astype(float)
    col2 = M[:, 2].astype(float)

    Naive_Bayes(col1, col2, labels, M)