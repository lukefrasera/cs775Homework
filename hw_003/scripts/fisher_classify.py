#!/usr/bin/env python
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import argparse
import os
import pdb
from scipy import spatial
import time
import operator

def ParseData(raw_data, class1, class2):
    raw_data = raw_data.rstrip('\n')
    raw_data_list = raw_data.split('\n')
    data_list = list()
    for raw_data_point in raw_data_list:
        raw_data_point = raw_data_point.rstrip()
        point = raw_data_point.split(' ')
        data_list.append([float(x) for x in point])
    data_list.pop()
    data_list_np = np.array(data_list)
    mask = (data_list_np[:,0] == class1) + (data_list_np[:,0] == class2)
    data_list_np = data_list_np[mask]
    return data_list_np

def FisherClassifier(features, classification, test_data, classa, classb):
    '''
    :param features:
    :param classification:
    :param test_data:
    :return:
    '''
    # separate classes
    class_a_features = features[classification == classa]
    class_b_features = features[classification == classb]

    class_a_mean = np.mean(class_a_features, 0).T
    class_a_cov  = np.cov(class_a_features.T)

    class_b_mean = np.mean(class_b_features, 0).T
    class_b_cov  = np.cov(class_b_features.T)

    # compute the Fisher criteria projection to one dimension
    project_a = la.inv(class_a_cov + class_b_cov) * (class_a_mean - class_b_mean)
    project_a = project_a / la.norm(project_a)

    # project all of the data
    class_a_project = class_a_features * project_a
    class_b_project = class_b_features * project_a





def main():
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('-t', '--training_file', type=str, help='submit data to train against')
    parser.add_argument('-f', '--testing_file', type=str, help='submit data to test the trained model against')
    parser.add_argument('-s', '--save_model', type=str, help='save out trained model')
    parser.add_argument('-r', '--read_model', type=str, help='read in trained model')
    parser.add_argument('-k', '--k_neighbors', type=int, help='number of neighbors to find')
    parser.add_argument('-a', '--classa', type=int, help='class to test/train on')
    parser.add_argument('-b', '--classb', type=int, help='class to test/train on')
    parser.add_argument('-m', '--method', type=int, help='0=KNN,1=LSE')

    args = parser.parse_args()
    print os.getcwd()
    # Check if Arguments allow execution
    if (not args.training_file) and (not args.read_model):
        print "Error: No training Data or model present!"
        return -1

    if args.training_file and args.read_model:
        print "Error: cannot read model and traing data at the same time!"
        return -1

    if args.training_file:
        # trainagainst training file
        if not os.path.isfile(args.training_file):
            print "Error: Training file doesn't exist!"
            return -1
        # train
        with open(args.training_file) as file:
            # read file contents
            raw_data = file.read()
            # parse data
        data = ParseData(raw_data, args.classa, args.classb)
        #plt.imshow(data[0,1:].reshape(1,256), cmap = plt.cm.Greys, interpolation = 'nearest')
        #plt.show()
        # train on data
        classification = data[:,0]
        features = np.matrix(data[:,1:])
    if args.testing_file:
        with open(args.testing_file) as test_file:
            raw_test_data = test_file.read()
            test_data = ParseData(raw_test_data, args.classa, args.classb)
            test_data_truth = test_data[:,0]
            test_data = np.matrix(test_data[:,1:])

    if args.method == 0:
        FisherClassifier(features, classification, test_data, args.classa, args.classb)

if __name__ == '__main__':
    main()