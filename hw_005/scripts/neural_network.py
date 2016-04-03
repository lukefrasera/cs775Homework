#!/usr/bin/env python
import csv, sys
from sympy import *
import numpy as np
import numexpr as ne
import pdb


class CSVInput:
    def __init__(self, filename, first_row_titles=False, num_convert=True, set_true_false_01=True):
        self.titles = []
        self.data = []
        self.boolean_false = ['F', 'f', 'False', 'FALSE', 'false']
        self.boolean_true  = ['T', 't', 'True', 'TRUE', 'true']
        with open(filename, 'rb') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i==0 and first_row_titles:
                    self.titles += row
                else:
                    if num_convert:
                        row_list = []
                        for elem in row:
                            try:
                                value = float(elem)
                            except ValueError:
                                try:
                                    value = int(elem)
                                except ValueError:
                                    value = elem
                                    if any(false in value for false in self.boolean_false):
                                      value = 0
                                    elif any(true in value for true in self.boolean_true):
                                      value = 1
                            row_list.append(value)
                    self.data.append(row_list)
        self.rows = len(self.data)
        self.cols = len(self.data[0])


class Sigmoid:
    def __init__(self):
        t = symbols('t')
        self.sigmoid = 1 / (1 + exp(-t))
        self.sigmoid_dif = self.sigmoid * (1 - self.sigmoid)

    def Sigmoid(self, t):
        return ne.evaluate(str(self.sigmoid))
    
    def SigmoidPrime(self, t):
        return ne.evaluate(str(self.sigmoid_dif))

class NeuralNetwork:
    def __init__(self, feature_size, compute_components, output_size):
        self.o_zero     = np.matrix(np.zeros((1, feature_size)))
        self.o_zero_bar = np.matrix(np.zeros((1, feature_size + 1)))

        self.W_one      = np.matrix(np.random.rand(feature_size, compute_components))
        self.W_one_bar  = np.matrix(np.random.rand(feature_size + 1, compute_components))

        self.o_one      = np.matrix(np.zeros((1, compute_components)))
        self.o_one_bar  = np.matrix(np.zeros((1, compute_components + 1)))

        self.W_two      = np.matrix(np.random.rand(compute_components, output_size))
        self.W_two_bar  = np.matrix(np.random.rand(compute_components + 1, output_size))
        self.S = Sigmoid()

    def FeedForward(self, feature, truth):
      # Compute the Feed forward pass of an iteration on the Neural Network
      # Need to append one onto the end of each of the weight and feature vector
      # pdb.set_trace()
      self.o_zero = np.matrix(feature)
      self.o_one = self.S.Sigmoid(self.o_zero * self.W_one)
      print self.o_one

    def BackProp(self):
      pass

    def UpdateWeights(self):
      pass


    
def main():
    # Read Data in and convert numbers to numbers
    reader = CSVInput(sys.argv[1], first_row_titles=True)

    # print reader.data
    neural = NeuralNetwork(reader.cols, 10, 1)

    x = np.matrix(np.ones((1, reader.cols)))
    t = np.matrix(np.ones((1, 1)))
    neural.FeedForward(x, t)

if __name__ == '__main__':
    main()
