#!/usr/bin/env python
import csv, sys
from sympy import *
import numpy as np
import numexpr as ne


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


class Sigmoid:
    def __init__(self):
        t = symbols('t')

        self.sigmoid = 1 / (1 + exp(-t))
        self.sigmoid_dif = self.sigmoid * (1 - self.sigmoid)
        # print self.sigmoid
        # print self.sigmoid_dif
    def Sigmoid(self, t):
        return ne.evaluate(str(self.sigmoid))
    
    def SigmoidPrime(self, t):
        return ne.evaluate(str(self.sigmoid_dif))

class NeuralNetwork:
    def __init__(self, feature_size, compute_components, output_size):
        self.o_zero = np.zeros(feature_size, 1)
        self.W_one = np.random(feature_size, compute_components)
        self.o_one = np.zeros(compute_components, 1)
        self.W_two = np.random(compute_components, output_size)
        self.S = Sigmoid()

    def FeedForward(self, feature, truth):
      # Compute the Feed forward pass of the first iteration on the Neural Network
      # Need to append one onto the end of each of the weight and feature vector
      self.o_zero = np.matrix(feature)
      self.o_one = self.S(self.o_zero * self.W_one)

    def BackProp(self):
      pass


    
def main():
    # Read Data in and convert numbers to numbers
    reader = CSVInput(sys.argv[1], first_row_titles=True)
    print reader.titles

    sigmoid = Sigmoid()
    print sigmoid.Sigmoid([4,5,6,7,8,9])
    print sigmoid.SigmoidPrime(np.array([4,5,6,7,8,9]))
    # print reader.data

if __name__ == '__main__':
    main()
