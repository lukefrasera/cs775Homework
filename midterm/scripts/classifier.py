#!/usr/bin/env python

import numpy as np
from numpy import linalg as la



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

    # solve for guassians

    # roject all of the data
    a_projected = self.projection * a_samples
    b_projected = self.projection * b_samples

    # genreate gaussian classifier
  def Classify(self, sample):
    pass
  def ReformatData(self, samples, truth):
    ref_samples = np.ones((len(samples)+1, len(samples[0])))
    ref_samples[:, 1:] = np.array(samples)

    ref_truth = np.array(truth)
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
    ref_samples = np.ones((len(smaples)+1, len(samples[0])))
    ref_samples[:, 1:] = np.array(samples)

    ref_truth = np.array(truth).T * 2 - 1
    return (ref_samples, ref_truth)

class Gaussian(Classifier):
  def __init__(self):
    pass
  def Train(self, samples, truth):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self):
    pass

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
  classifier = Regression()

  testing = ClassifierTest(classifier, training_set, testing_set)

  testing.Training()
  testing.Testing()
  results = testing.Results()

  GraphResults(results)
  GenerateTable(results)

if __name__ == '__main__':
  main()
