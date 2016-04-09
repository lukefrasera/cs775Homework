#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import numexpr as ne
import sys, csv

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
    pass
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
    a_samples = samples[truth == self.c_a]
    b_samples = samples[truth == self.c_b]

    # ompute mean and covariance
    a_mean = np.mean(a_samples, 0).T
    b_mean = np.mean(b_samples, 0).T

    a_cov  = np.cov(a_samples.T)
    b_cov  = np.cov(b_samples.T)

    # Compute fisher criteria projection to one dimension
    self.projection = la.inv(a_cov + b_cov) * (a_mean - b_mean)
    self.projection /= la.norm(sel.projection)

    # project all of the data
    a_projected = self.projection * a_samples
    b_projected = self.projection * b_samples

    # genreate gaussian classifier
    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    self.a_guass.Train(a_projected)
    self.b_guass.Train(b_projected)

  def Classify(self, samples):
    # project samples into space
    projected = self.projection * samples

    # Perform Gaussian classification
    a_prob = self.a_gauss.Classify(projected)
    b_prob = self.b_gauss.Classify(projected)
    a = self.c_a
    b = self.c_b

    # classify against probability
    return ne.evaluate('where(a_prob > b_prob, a, b)')

  def ReformatData(self, samples, truth):
    ref_samples = np.ones((len(samples), len(samples[0])+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth)
    return (ref_samples, ref_truth)

class Regression(Classifier):
  def __init__(self):
    self.a = []

  def Train(self, samples, truth):
    samples = np.matrix(samples)
    truth = np.matrix(truth)
    self.a = la.inv(samples.T * smples) * smples.T * truth

  def Classify(self, samples):
    samples = np.matrix(samples)
    return samples * self.a

  def ReformatData(self, samples, truth):
    ref_samples = np.ones((len(samples), len(samples[0])+1))
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
    self.cov = np.cov(samples)
    self.cov_inv = self.cov.I

    # Compute normalizing term
    self.normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * self.cov)))

  def ClassifySample(self, sample):
    return self.normalizer * np.exp(- 0.5 * (sample - self.mean).T * self.cov_inv * (sample - self.mean))

  def Classify(self, samples):
    # compute mahalanobis distance
    dist = self.cov_inv * samples
    dist = np.multiply(samples, dist)
    dist = np.sum(dist, 0)
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
    self.projection = np.random(1, len(samples[0]))

    # project training samples
    a_samples = samples[truth == self.c_a]
    b_samples = samples[truth == self.c_b]

    a_projected = self.projection * a_samples
    b_projected = self.projection * b_samples

    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    self.a_gauss.Train(a_projected)
    self.b_gauss.Train(b_projected)

  def Classify(self, samples):
    projected = self.projection * samples

    a = self.c_a
    b = selg.c_b

    a_prob = self.a_gauss.Classify(projected)
    b_prob = self.b_gauss.Classify(projected)

    return ne.evaluate('where(a_prob > b_prob, a, b)')


  def ReformatData(self, samples, truth):
    ref_samples = np.ones((len(samples), len(samples[0])+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth)
    return (ref_samples, ref_truth)

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
  samples = CSVInput(sys.argv[1], first_row_titles=False)
  truth = CSVInput(sys.argv[2], first_row_titles=False)

  regression = Regression()
  gaussian = Gaussian()
  fisher = Fisher(0, 1)
  random = Random(0, 1)

  reg_samples, reg_truth = regression.ReformatData(samples.data, truth.data)
  # print reg_samples
  # print reg_truth

  gauss_samples = gaussian.ReformatData(samples.data)

  # print gauss_samples

  fish_samples, fish_truth = fisher.ReformatData(samples.data, truth.data)
  print fish_samples
  print fish_truth

  # testing = ClassifierTest(classifier, training_set, testing_set)

  # testing.Training()
  # testing.Testing()
  # results = testing.Results()

  # GraphResults(results)
  # GenerateTable(results)

if __name__ == '__main__':
  main()
