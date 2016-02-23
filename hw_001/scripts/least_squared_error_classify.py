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
'''
Python Program demonstrating the use of a LSE (Least Squared Error) and KNN (K Nearest Neighbors) classifier.
'''

#KNNCLassifier returns a tuple of the K closest feature vectors
def KNNSearch(k, features, test_data):
    test_data_classification = []
    for test_index, test_element in enumerate(test_data):
        if test_element == []:
            continue
        neighborDistance = []
        for feature_index,feature in enumerate(features):
            try:
                distance = la.norm(feature-test_element)
            except ValueError:
                pdb.set_trace()
            neighborDistance.append([distance, feature_index])
        neighborDistance = sorted(neighborDistance, key=lambda row: row[0], reverse=True)
        #pdb.set_trace()
        test_data_classification.append(np.matrix(neighborDistance[0:k][1]))
    pdb.set_trace()
    return test_data_classification
    
def KNNSearchFast(k, features, test_data):
    t0 = time.time()
    tree = spatial.KDTree(features)
    t1 = time.time()
    result = tree.query(test_data, k)
    t2 = time.time()
    print "Build time: %f, query time: %f" % (t1-t0, t2-t1)
    return result
    
def KNNClassify(train_classification, test_neighbors):
    test_classification = []
    for sample in test_neighbors[1]:
        votes = [0 for x in xrange(10)]
        try:
            for neighbor in sample:
                sample_class = int(train_classification[neighbor])
                votes[sample_class] += 1
        except TypeError:
            #catch the case where K=1
            sample_class = int(train_classification[sample])
            votes[sample_class] = 1
        classification = max(enumerate(votes), key=operator.itemgetter(1))[0]
        test_classification.append(classification)
    return test_classification

def LSESearch(features,classification, test_data):
     features = np.matrix(features)
     classification = np.matrix(classification).T
     test_data = np.matrix(test_data)
     filter = la.inv(features.T * features)  * features.T * classification
     test_data_classification = []
     classification = (test_data * filter)
     classification[classification < 0] = -1
     classification[classification >=0] = 1
     return classification

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
				#Do KNN classification
        nearest_neighbors = KNNSearchFast(args.k_neighbors, features, test_data)
        print "Num training samples: %d, num test samples: %d" % (len(classification), len(test_data_truth))
        classification = KNNClassify(classification, nearest_neighbors)

				#Compute the error rate
        errors = test_data_truth - classification
        misclassification_a = errors[errors == args.classa - args.classb]
        misclassification_b = errors[errors == args.classb - args.classa]
        mask = errors != 0
        num_errors = sum(mask)
        print "Error rate: %f%%" % (float(num_errors)/len(test_data_truth)*100)
        print "Percentage of %d's misclassified: %f" % (args.classa, 
														float(misclassification_a.size)/test_data_truth[test_data_truth == args.classa].size*100)
        print "Percentage of %d's misclassified: %f" % (args.classb, float(misclassification_b.size)/test_data_truth[test_data_truth ==  args.classb].size*100)
    if args.method == 1:
				#Do LSE classification
        #make classification binary
        classification[classification == args.classa] = -1
        classification[classification == args.classb] = 1

				#Perform the classficiation on the test data
        test_data_classification = LSESearch(features, classification, test_data)
        test_data_truth[test_data_truth == args.classa] = -1
        test_data_truth[test_data_truth == args.classb] = 1

				#Compute the error rate
        errors = test_data_classification.T - np.matrix(test_data_truth)
        misclassification_a = errors[errors == 2]
        misclassification_b = errors[errors == -2]
        num_errors = np.sum(np.absolute(errors))
        print "Num training samples: %d, num test samples: %d" % (len(classification), len(test_data_truth))
        print "Error rate: %f%%" % (float(num_errors)/len(test_data_truth)*100)
        print "Percentage of %d's misclassified: %f" % (args.classa, float(misclassification_a.size)/test_data_truth[test_data_truth == -1].size*100)
        print "Percentage of %d's misclassified: %f" % (args.classb, float(misclassification_b.size)/test_data_truth[test_data_truth ==  1].size*100)
if __name__ == '__main__':
    main()
