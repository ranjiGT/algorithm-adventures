import numpy as np
import csv
from numpy import genfromtxt
import argparse


def nnet(X_train, Y_train, iter, step_size):

    """passive weights for Gauss3 and Gauss4"""
    w_a_h1 = -0.3
    w_b_h1 = 0.4
    w_a_h2 = -0.1
    w_b_h2 = -0.4
    w_a_h3 = 0.2
    w_b_h3 = 0.1
    w_h1_o = 0.1
    w_h2_o = 0.3
    w_h3_o = -0.4
    w_bias_h1 = 0.2
    w_bias_h2 = -0.5
    w_bias_h3 = 0.3
    w_bias_o = -0.1

    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")

    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")
    print('-', end=" ")

    print(round(w_bias_h1, 5), end=" ")
    print(round(w_a_h1, 5), end=" ")
    print(round(w_b_h1, 5), end=" ")

    print(round(w_bias_h2, 5), end=" ")
    print(round(w_a_h2, 5), end=" ")
    print(round(w_b_h2, 5), end=" ")

    print(round(w_bias_h3, 5), end=" ")
    print(round(w_a_h3, 5), end=" ")
    print(round(w_b_h3, 5), end=" ")

    print(round(w_bias_o, 5), end=" ")
    print(round(w_h1_o, 5), end=" ")
    print(round(w_h2_o, 5), end=" ")
    print(round(w_h3_o, 5))


    for _ in range(iter):
        for i in range(len(X_train)):
            
            '''Observable output from sigmoid unit'''
            net_h1 = w_a_h1*X_train[i][0] + w_b_h1*X_train[i][1] + w_bias_h1 
            out_h1 = 1/(1+np.exp(-net_h1))

            net_h2 = w_a_h2*X_train[i][0] + w_b_h2*X_train[i][1] + w_bias_h2
            out_h2 = 1/(1+np.exp(-net_h2))

            net_h3 = w_a_h3*X_train[i][0] + w_b_h3*X_train[i][1] + w_bias_h3
            out_h3 = 1/(1+np.exp(-net_h3))

            net_o = out_h1 *w_h1_o + out_h2*w_h2_o + out_h3*w_h3_o + w_bias_o
            out = 1/(1+np.exp(-net_o))
            
            
            error = (Y_train[i]-out) #(t-o)
            
            '''Chain rule'''
            delta_o = out*(1-out)*error
            delta_h1 = out_h1*(1-out_h1)*(delta_o*w_h1_o)
            delta_h2 = out_h2*(1-out_h2)*(delta_o*w_h2_o)
            delta_h3 = out_h3*(1-out_h3)*(delta_o*w_h3_o)
    
            w_h1_o = w_h1_o + step_size*delta_o*out_h1
            

            w_h2_o = w_h2_o + step_size*delta_o*out_h2
            

            w_h3_o = w_h3_o + step_size*delta_o*out_h3
            

            w_bias_o = w_bias_o + step_size*delta_o
            

            '''Active node-Information flow 1'''          
            w_a_h1 = w_a_h1 + step_size*delta_h1*X_train[i][0]
            
            w_b_h1 = w_b_h1 + step_size*delta_h1*X_train[i][1]
            
            w_bias_h1 = w_bias_h1 + step_size*delta_h1
            
            
            '''Active node-Information flow 2'''
            w_a_h2 = w_a_h2 + step_size*delta_h2*X_train[i][0]
            
            w_b_h2 = w_b_h2 + step_size*delta_h2*X_train[i][1]
            
            w_bias_h2 = w_bias_h2 + step_size*delta_h2
            
            
            '''Active node-Information flow 3'''
            w_a_h3 = w_a_h3 + step_size*delta_h3*X_train[i][0]
            
            w_b_h3 = w_b_h3 + step_size*delta_h3*X_train[i][1]
            
            w_bias_h3 = w_bias_h3 + step_size*delta_h3
            

    
            print(X_train[i][0], end=" ")
            print(X_train[i][1], end=" ")
            print(round(out_h1, 5), end=" ")
            print(round(out_h2, 5), end=" ")
            print(round(out_h3, 5), end=" ")
            print(round(out, 5), end=" ")
            print(int(Y_train[i]), end=" ")

            print(round(delta_h1, 5), end=" ")
            print(round(delta_h2, 5), end=" ")
            print(round(delta_h3, 5), end=" ")
            print(round(delta_o, 5), end=" ")

            print(round(w_bias_h1, 5), end=" ")
            print(round(w_a_h1, 5), end=" ")
            print(round(w_b_h1, 5), end=" ")

            print(round(w_bias_h2, 5), end=" ")
            print(round(w_a_h2, 5), end=" ")
            print(round(w_b_h2, 5), end=" ")

            print(round(w_bias_h3, 5), end=" ")
            print(round(w_a_h3, 5), end=" ")
            print(round(w_b_h3, 5), end=" ")

            print(round(w_bias_o, 5), end=" ")
            print(round(w_h1_o, 5), end=" ")
            print(round(w_h2_o, 5), end=" ")
            print(round(w_h3_o, 5))


"""Diver code"""

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                       help='Gauss files as csv')

    parser.add_argument('--eta',
                       help='learning rate')

    parser.add_argument('--iterations',
                       help='threshold value')

    args = parser.parse_args()

    file_path = args.data
    step_size = float(args.eta)
    iter = int(args.iterations)


    x = genfromtxt(file_path, delimiter=',', autostrip=True)
    M = np.array(x).astype(float)
    M = np.round(M, 5)
    Y_train = M[:, -1].astype(float)
    X_train = M[:, :-1].astype(float)

    nnet(X_train, Y_train, iter, step_size)    