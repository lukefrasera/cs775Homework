#!/usr/bin/env python

import numpy as np



class Classifier(object):
  def __init__(self):
    pass
  def Train(self, samples):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self):
    pass

class Fisher(Classifier):
  def __init__(self):
    pass
  def Train(self, samples):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self):
    pass

class Regression(Classifier):
  def __init__(self):
    pass
  def Train(self, samples):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self):
    pass

class Gaussian(Classifier):
  def __init__(self):
    pass
  def Train(self, samples):
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
