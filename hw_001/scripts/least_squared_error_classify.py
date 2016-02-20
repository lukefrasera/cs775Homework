#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
'''
Python Program demonstrating the use of a LSE (Least Squared Error) classifier.
'''

def ParseData(raw_data):
    raw_data_list = raw_data.split('\n')
    data_list = list()
    for raw_data_point in raw_data_list:
        point = raw_data_point.split(' ')
        point.pop()
        data_list.append([float(x) for x in point])
    data_list.pop()
    return np.array(data_list)


def main():

    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('-t', '--training_file', type=str, help='submit data to train against')
    parser.add_argument('-f', '--testing_file', type=str, help='submit data to test the trained model against')
    parser.add_argument('-s', '--save_model', type=str, help='save out trained model')
    parser.add_argument('-r', '--read_model', type=str, help='read in trained model')
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
        data = ParseData(raw_data)

        # train on data
        
            


if __name__ == '__main__':
    main()