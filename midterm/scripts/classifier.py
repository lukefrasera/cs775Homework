#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import numexpr as ne
import sys, csv
import pdb

class CSVInput:
  def __init__(self, filename, first_row_titles=False, num_convert=True, set_true_false_01=True):
    self.titles = []
    self.data = []
    self.boolean_false = ['F', 'f', 'False', 'FALSE', 'false']
    self.boolean_true  = ['T', 't', 'True', 'TRUE', 'true']
    with open(filename, 'rb') as file:
      reader = csv.reader(file, delimiter=' ')
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

class Classifier(object):
  def __init__(self):
    pass
  def Train(self, samples, truth):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self):
    return (samples, truth)

class Fisher(Classifier):
  def __init__(self, class_a, class_b):
    self.projection = []
    self.c_a = class_a
    self.c_b = class_b
    self.a_gauss = 0
    self.b_gauss = 0

  def Train(self, samples, truth):
    # solve for projection
    a_samples = np.asmatrix(samples[np.asarray(truth.T)[0] == self.c_a])
    b_samples = np.asmatrix(samples[np.asarray(truth.T)[0] == self.c_b])

    # ompute mean and covariance
    a_mean = np.mean(a_samples, 0).T
    b_mean = np.mean(b_samples, 0).T

    a_cov  = np.cov(a_samples.T)
    b_cov  = np.cov(b_samples.T)

    # Compute fisher criteria projection to one dimension
    self.projection = la.inv((a_cov + b_cov) + np.eye(a_cov.shape[0]) * 0.00001) * (a_mean - b_mean)
    self.projection /= la.norm(self.projection)

    # project all of the data
    a_projected = a_samples * self.projection
    b_projected = b_samples * self.projection

    # genreate gaussian classifier
    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    self.a_gauss.Train(a_projected)
    self.b_gauss.Train(b_projected)

  def Classify(self, samples):
    # project samples into space
    projected = samples * self.projection

    # Perform Gaussian classification
    a_prob = self.a_gauss.Classify(projected)
    b_prob = self.b_gauss.Classify(projected)
    a = self.c_a
    b = self.c_b

    # classify against probability
    return ne.evaluate('where(a_prob > b_prob, a, b)')

  def ReformatData(self, samples, truth):
    ref_samples = np.ones((samples.shape[0], samples.shape[1]+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth)
    return (ref_samples, ref_truth)

class Regression(Classifier):
  def __init__(self, class_a, class_b):
    self.a = []
    self.c_a = class_a
    self.c_b = class_b

  def Train(self, samples, truth):
    samples = np.matrix(samples)
    truth = np.matrix(truth)
    self.a = la.inv(samples.T * samples) * samples.T * truth

  def Classify(self, samples):
    samples = np.matrix(samples)
    projection = samples * self.a
    result = np.zeros(projection.shape)

    result[projection < 0] = self.c_a
    result[projection >=0] = self.c_b

    return result

  def ReformatData(self, samples, truth):
    ref_samples = np.ones((samples.shape[0], samples.shape[1]+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth) * 2 - 1
    return (ref_samples, ref_truth)

class Gaussian(Classifier):
  def __init__(self):
    self.cov_inv = []
    self.mean = []
    self.normalizer = []
  def Train(self, samples):
    self.mean = np.mean(samples, 0).T
    self.cov = np.cov(samples.T)
    if self.cov.shape != ():
      self.cov_inv = la.inv(self.cov)
    else:
      self.cov_inv = 1.0 / self.cov

    # Compute normalizing term
    if self.cov.shape != ():
      self.normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * self.cov)))
    else:
      self.normalizer = 1.0 / (np.sqrt(2.0 * np.pi * self.cov))

  def ClassifySample(self, sample):
    return self.normalizer * np.exp(- 0.5 * (sample - self.mean).T * self.cov_inv * (sample - self.mean))

  def Classify(self, samples):
    # compute mahalanobis distance
    dist = self.cov_inv * samples.T
    dist = np.multiply(samples.T, dist)
    dist = np.sum(dist, 0).T
    # compute exponent
    return self.normalizer * np.exp(-0.5 * dist)

  def ReformatData(self, samples):
    return np.matrix(samples)


class Random(Classifier):
  def __init__(self, class_a, class_b):
    self.projection = 0
    self.a_gauss = 0
    self.b_gauss = 0
    self.c_a = class_a
    self.c_b = class_b

  def Train(self, samples, truth):
    # randomly select projection
    self.projection = np.random.rand(samples.shape[1], 1)
    self.projection /= la.norm(self.projection)

    # pdb.set_trace()
    # project training samples
    a_samples = np.asmatrix(samples[np.asarray(truth.T)[0] == self.c_a])
    b_samples = np.asmatrix(samples[np.asarray(truth.T)[0] == self.c_b])

    # pdb.set_trace()
    a_projected = a_samples * self.projection
    b_projected = b_samples * self.projection

    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    self.a_gauss.Train(a_projected)
    self.b_gauss.Train(b_projected)

  def Classify(self, samples):
    # project samples into space
    # pdb.set_trace()
    projected = samples * self.projection

    # Perform Gaussian classification
    a_prob = self.a_gauss.Classify(projected)
    b_prob = self.b_gauss.Classify(projected)
    a = self.c_a
    b = self.c_b

    # classify against probability
    return ne.evaluate('where(a_prob > b_prob, a, b)')


  def ReformatData(self, samples, truth):
    ref_samples = np.ones((samples.shape[0], samples.shape[1]+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth)
    return (np.asmatrix(ref_samples), ref_truth)

class ClassiferTest(object):
  def __init__(self, classifier, training_set, testing_set):
    self.classifier   = classifier
    self.training_set = training_set
    self.testing_set  = testing_set

  def Training(self):
    pass
  def Testing(self):
    pass
  def Results(self):
    pass

def GraphResults(results):
  pass

def GenerateTable(results):
  pass

def main():
  ''' Test the classes for performance and corrrecness'''
  data = CSVInput(sys.argv[1], first_row_titles=False)
  # truth = CSVInput(sys.argv[2], first_row_titles=False)
  samples = np.matrix(data.data)
  truth = samples[:,-1]
  samples = samples[:,:-1]

  # print samples, samples.shape
  # print truth, truth.shape


  regression = Regression(0, 1)
  reg_samples, reg_truth = regression.ReformatData(samples, truth)
  regression.Train(reg_samples, reg_truth)
  reg_result = regression.Classify(reg_samples)
  compare = reg_result - ((reg_truth + 1.0)/ 2.0)
  compare = compare != 0
  print "Regression:"
  print float(np.sum(compare)) / float(compare.shape[0])
  print compare.shape
  
  fisher = Fisher(0, 1)
  fish_samples, fish_truth = fisher.ReformatData(samples, truth)
  fisher.Train(fish_samples, fish_truth)
  fish_result = fisher.Classify(fish_samples)

  compare = fish_result - fish_truth
  compare = compare != 0
  print "Fisher"
  print float(np.sum(compare)) / float(compare.shape[0])
  print compare.shape
  
  random = Random(0, 1)
  rand_samples, rand_truth = random.ReformatData(samples, truth)
  random.Train(rand_samples, rand_truth)
  rand_result = random.Classify(rand_samples)

  compare = rand_result - rand_truth
  compare = compare != 0

  print "Random"
  print float(np.sum(compare)) / float(compare.shape[0])
  print compare.shape

  # testing = ClassifierTest(classifier, training_set, testing_set)

  # testing.Training()
  # testing.Testing()
  # results = testing.Results()

  # GraphResults(results)
  # GenerateTable(results)

if __name__ == '__main__':
  main()
