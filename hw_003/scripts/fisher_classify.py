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
    mask = (data_list_np[:, 0] == class1) + (data_list_np[:, 0] == class2)
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

    class_a_gauss_build = GaussianBuild(class_a_project)
    class_b_gauss_build = GaussianBuild(class_b_project)

    # class_a_prob = []
    # class_b_prob = []
    classification_result = []
    for sample in test_data:
        sample_project = sample * project_a
        class_a_prob = ComputeGaussianProbability(class_a_gauss_build[0], class_a_gauss_build[1], sample_project)
        class_b_prob = ComputeGaussianProbability(class_b_gauss_build[0], class_b_gauss_build[1], sample_project)
        if class_a_prob > class_b_prob:
            classification_result.append(classa)
        else:
            classification_result.append(classb)
    return classification_result

def GaussianBuild(features):
    """
    computes the mean and covariance for a dataset
    :param features: s x f np.matrix (s samples by f features)
    :param classification: s x 1 np.ndarray
    :param class_id: scalar value to find
    :return: [covariance(f x f),mean (f x 1)]
    """
    #pdb.set_trace()
    print 'Of ', features.shape, 'Elements, ', features.shape
    cov_mat = np.cov(features.T)
    mean_mat = np.mean(features.T)
    return [cov_mat, mean_mat]


def ComputeGaussianProbability(cov_mat, mean_mat, sample):
    """
    computes the probability of a particular sample belonging to a particular gaussian distribution
    :param cov_mat: f x f np.matrix (f features)
    :param mean_mat: f x 1 np.matrix
    :param sample: f x 1 np.matrix
    :return:
    """
    mean_mat = np.matrix(mean_mat).T
    sample = sample.T
    # sample = meanMat
    non_invertible = True
    eye_scale = 0.0
    cov_mat_inverse = 1.0 / cov_mat
    probability = 1.0 / (np.sqrt(la.norm(2 * np.pi * cov_mat)))
    probability *= np.exp(-0.5 * (sample - mean_mat).T * cov_mat_inverse * (sample - mean_mat))
    return probability

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
        # plt.imshow(data[0,1:].reshape(1,256), cmap = plt.cm.Greys, interpolation = 'nearest')
        # plt.show()
        # train on data
        classification = data[:, 0]
        features = np.matrix(data[:, 1:])
    if args.testing_file:
        with open(args.testing_file) as test_file:
            raw_test_data = test_file.read()
            test_data = ParseData(raw_test_data, args.classa, args.classb)
            test_data_truth = test_data[:, 0]
            test_data = np.matrix(test_data[:, 1:])

    if args.method == 0:
        result = FisherClassifier(features, classification, test_data, args.classa, args.classb)
        print result
        print [int(x) for x in list(test_data_truth)]
        errors = np.array(result) == test_data_truth
        class_a_samples = errors[test_data_truth == args.classa]
        class_b_samples = errors[test_data_truth == args.classb]
        num_a_correct = np.sum(class_a_samples)
        num_b_correct = np.sum(class_b_samples)
        total_a = class_a_samples.shape[0]
        total_b = class_b_samples.shape[0]
        print (1.0-float(num_a_correct)/total_a)*100,'%% of class a was misclassified'
        print (1.0-float(num_b_correct)/total_b)*100,'%% of class b was misclassified'



if __name__ == '__main__':
    main()

