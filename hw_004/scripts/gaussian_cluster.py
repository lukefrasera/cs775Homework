#!/usr/bin/env python
import numpy as np
from math import floor
from numpy import linalg as la
from numpy import matlib as matlib
import matplotlib.pyplot as plt
import argparse
import os
import pdb
from scipy import spatial
import time
import operator
import random


def ParseData(raw_data):
    raw_data = raw_data.rstrip('\n')
    raw_data_list = raw_data.split('\n')
    data_list = list()
    for raw_data_point in raw_data_list:
        raw_data_point = raw_data_point.rstrip()
        point = raw_data_point.split(' ')
        data_list.append([float(x) for x in point])
    data_list.pop()
    data_list_np = np.array(data_list)
    return data_list_np

def ComputeMahalanobisDistance(sample, covariance):
    invertible = False
    lambda_scale = 10e-10
    while not invertible:
        try:
            right = la.inv(covariance + np.eye(covariance.shape[0])*lambda_scale)*sample.T
        except la.LinAlgError:
            lambda_scale += 10e-10
            continue
        invertible = True
    left = np.multiply(sample.T,right)
    distance = np.sqrt(np.sum(left,0))
    return distance

def GaussianCluster(train_data, test_data, iterations, num_clusters):
    #general algorithm: use the elements in features to construct the class separations
    #then test the separations on the test data

    #first, select random means for each cluster
    num_features = train_data.shape[1]
    num_train_samples = train_data.shape[0]
    means = np.zeros((num_clusters,num_features))
    covariance = np.zeros((num_clusters, num_features, num_features))
    labels = np.zeros(num_train_samples)
    for cluster in range(num_clusters):
        sample_index = floor(random.random() * num_train_samples)
        means[cluster] = train_data[sample_index]
        covariance[cluster] = np.eye(num_features)
    #Now for each iteration, check find the closest cluster and assign the data point to that cluster
    sample_mdistances = np.zeros([num_clusters,num_train_samples])
    for iteration in range(iterations):
        #assign all the data to a class
        for cluster_index in range(num_clusters):
            sample_distance = np.subtract(train_data,means[cluster_index])
            sample_mdistances[cluster_index] = ComputeMahalanobisDistance(sample_distance,covariance[cluster_index])
        #assign a label to each sample
        labels = np.argmin(sample_mdistances,0)
        #recompute the mean and covariance of each cluster
        for cluster_index in range(num_clusters):
            mask = labels == cluster_index
            this_cluster = train_data[mask,:]
            means[cluster_index] = np.mean(this_cluster,0)
            covariance[cluster_index,:,:] = np.cov(this_cluster.T)
    return [labels,means,covariance]

def main():
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('-t', '--training_file', type=str, help='submit data to train against')
    parser.add_argument('-f', '--testing_file', type=str, help='submit data to test the trained model against')
    parser.add_argument('-s', '--save_model', type=str, help='save out trained model')
    parser.add_argument('-r', '--read_model', type=str, help='read in trained model')
    parser.add_argument('-m', '--method', type=int, help='0=GaussCluster')
    parser.add_argument('-c', '--clusters', type=int, help='num clusters')
    parser.add_argument('-i', '--iters', type=int, help='iterations to run')

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
        data = ParseData(raw_data)
        # plt.imshow(data[0,1:].reshape(1,256), cmap = plt.cm.Greys, interpolation = 'nearest')
        # plt.show()
        # train on data
        classification = data[:, 0]
        features = np.matrix(data[:, 1:])
    if args.testing_file:
        with open(args.testing_file) as test_file:
            raw_test_data = test_file.read()
            test_data = ParseData(raw_test_data)
            test_data_truth = test_data[:, 0]
            test_data = np.matrix(test_data[:, 1:])

    if args.method == 0:
        beginTime = time.clock()
        [train_labels,means,covariance] = GaussianCluster(features, test_data, args.iters, args.clusters)
        endTime = time.clock()
        print "Time elapsed "+str(endTime-beginTime)


        #Run the clustering algorithm on the test data set
        num_test_samples = test_data.shape[0]
        sample_mdistances = np.zeros([args.clusters,num_test_samples])
        for cluster_index in range(args.clusters):
            sample_distance = np.subtract(test_data,means[cluster_index])
            sample_mdistances[cluster_index] = ComputeMahalanobisDistance(sample_distance,covariance[cluster_index])
        #assign a label to each sample
        test_labels = np.argmin(sample_mdistances,0)

        #now compute the composition of each cluster
        histogram_database = np.zeros((args.clusters,args.clusters))
        for cluster_index in range(args.clusters):
            mask = test_labels == cluster_index
            this_cluster = test_data_truth[mask]
            histogram = np.histogram(this_cluster,bins=range(args.clusters+1))[0]
            histogram_database[cluster_index] = histogram
        total_samples = np.sum(histogram_database,0)
        temp = [["%6d" % j for j in list(i)] for i in list(histogram_database)]
        for item in temp:
            print item
        normalized_histogram = np.divide(histogram_database,matlib.repmat(total_samples,args.clusters,1))
        normalized_histogram = [["%6.2f" % j for j in list(i)] for i in list(normalized_histogram)]
        for hist in normalized_histogram:
            print hist
if __name__ == '__main__':
    main()

