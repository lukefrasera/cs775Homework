#!/usr/bin/env python

import numpy as np
from numpy import linalg as la
import numexpr as ne
import sys, csv
import pdb
import copy
import warnings
import random
import matplotlib.pyplot as plt

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
  def ReformatData(self, samples, truth):
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

    # compute mean and covariance
    a_mean = np.asmatrix(np.mean(a_samples, 0).T)
    b_mean = np.asmatrix(np.mean(b_samples, 0).T)

    a_cov  = np.asmatrix(np.cov(a_samples.T))
    b_cov  = np.asmatrix(np.cov(b_samples.T))

    # Compute fisher criteria projection to one dimension
    if a_samples.shape[0] == 0:
      a_cov = np.zeros(b_cov.shape)
      a_mean = np.zeros(b_mean.shape)
      error = True
    if b_samples.shape[0] == 0:
      b_cov = np.zeros(a_cov.shape)
      b_mean = np.zeros(a_mean.shape)
      error = True
    self.projection = la.inv((a_cov + b_cov) + np.eye(a_cov.shape[0]) * 0.00001) * (a_mean - b_mean)
    self.projection /= la.norm(self.projection)

    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    # project all of the data
    if a_samples.shape[0] != 0:
      a_projected = a_samples * self.projection
      self.a_gauss.Train(a_projected)
    else:
      self.a_gauss = None
    if b_samples.shape[0] != 0:
      b_projected = b_samples * self.projection
      self.b_gauss.Train(b_projected)
    else:
      self.b_gauss = None

  def Classify(self, samples):
    # project samples into space
    projected = samples * self.projection

    # Perform Gaussian classification
    if self.a_gauss:
      a_prob = self.a_gauss.Classify(projected)
    else:
      a_prob = np.zeros((projected.shape[0], 1))
    if self.b_gauss:
      b_prob = self.b_gauss.Classify(projected)
    else:
      b_prob = np.zeros((projected.shape[0], 1))
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
    truth = np.matrix(truth * 2 - 1)
    try:
      self.a = la.inv(samples.T * samples) * samples.T * truth
    except la.linalg.LinAlgError:
      self.a = la.inv(samples.T * samples + np.eye(samples.shape[1]) * 0.0000001) * samples.T * truth

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

    ref_truth = np.matrix(truth)
    return (np.asmatrix(ref_samples), ref_truth)

class Gaussian(Classifier):
  def __init__(self):
    self.cov_inv = []
    self.mean = []
    self.normalizer = []
  def Train(self, samples):
    self.mean = np.mean(samples, 0).T
    self.cov = np.cov(samples.T)
    if samples.shape[0] == 1:
      self.cov = np.ones(self.cov.shape)
    if self.cov.shape != ():
      self.cov_inv = la.inv(self.cov)
    else:
      self.cov_inv = 1.0 / (self.cov + 0.000000001)

    # Compute normalizing term
    if self.cov.shape != ():
      self.normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * self.cov)))
    else:
      self.normalizer = 1.0 / (np.sqrt(2.0 * np.pi * (self.cov + 0.000000001)))

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
    # a_projected = a_samples * self.projection
    # b_projected = b_samples * self.projection

    self.a_gauss = Gaussian()
    self.b_gauss = Gaussian()

    # self.a_gauss.Train(a_projected)
    # self.b_gauss.Train(b_projected)

    if a_samples.shape[0] != 0:
      a_projected = a_samples * self.projection
      self.a_gauss.Train(a_projected)
    else:
      self.a_gauss = None
    if b_samples.shape[0] != 0:
      b_projected = b_samples * self.projection
      self.b_gauss.Train(b_projected)
    else:
      self.b_gauss = None

  def Classify(self, samples):
    # project samples into space
    # pdb.set_trace()
    # projected = samples * self.projection
    projected = samples * self.projection

    # Perform Gaussian classification
    if self.a_gauss:
      a_prob = self.a_gauss.Classify(projected)
    else:
      a_prob = np.zeros((projected.shape[0], 1))
    if self.b_gauss:
      b_prob = self.b_gauss.Classify(projected)
    else:
      b_prob = np.zeros((projected.shape[0], 1))
    # Perform Gaussian classification
    # a_prob = self.a_gauss.Classify(projected)
    # b_prob = self.b_gauss.Classify(projected)
    a = self.c_a
    b = self.c_b

    # classify against probability
    return ne.evaluate('where(a_prob > b_prob, a, b)')


  def ReformatData(self, samples, truth):
    ref_samples = np.ones((samples.shape[0], samples.shape[1]+1))
    ref_samples[:, 1:] = np.matrix(samples)

    ref_truth = np.matrix(truth)
    return (np.asmatrix(ref_samples), ref_truth)

################################################################################
# Decision Tree Classifier
################################################################################
class Node(object):
  def __init__(self, classifier):
    self.classifier = copy.deepcopy(classifier)
    self.left  = None
    self.right = None

class DecisionTree(Classifier):
  def __init__(self, classifier, class_a, class_b, max_depth=1000):
    self.tree = Node(copy.deepcopy(classifier))
    self.classifier = classifier
    self.max_depth = max_depth
  def ReformatData(self, samples, truth):
    return self.classifier.ReformatData(samples, truth)
  def Train(self, samples, truth):
    # pdb.set_trace()
    self.TrainRecur(self.tree, samples, truth, 1)
  def TrainRecur(self, node, samples, truth, depth):

    node.classifier.Train(samples, truth)
    if depth > self.max_depth:
      return
    result = node.classifier.Classify(samples)

    compare   = result != truth
    a_compare = np.sum(compare[result.T[0] == 0])
    b_compare = np.sum(compare[result.T[0] == 1])

    # pdb.set_trace()
    if a_compare > 0:
      a_samples  = samples[result.T[0] == 0]
      if not np.array_equal(a_samples, samples):
        a_truth    = truth[result.T[0] == 0]
        node.left  = Node(self.classifier)
        self.TrainRecur(node.left, a_samples, a_truth, depth + 1)
      # a_truth = truth[result.T[0] == 0]
      # node.left = Node(self.classifier)
      # self.TrainRecur(node.left, a_samples, a_truth, depth + 1)

    if b_compare > 0:
      b_samples  = samples[result.T[0] == 1]
      if not np.array_equal(b_samples, samples):
        b_truth    = truth[result.T[0] == 1]
        node.right = Node(self.classifier)
        self.TrainRecur(node.right, b_samples, b_truth, depth + 1)
      # b_truth = truth[result.T[0] == 1]
      # node.right = Node(self.classifier)
      # self.TrainRecur(node.right, b_samples, b_truth, depth + 1)

  def Classify(self, samples):
    return self.ClassifyRecur(self.tree, samples)

  def ClassifyRecur(self, node, samples):
    result = node.classifier.Classify(samples)
    output = np.zeros(result.shape)
    if node.left:
      a_samples = samples[result.T[0] == 0]
      a_result = self.ClassifyRecur(node.left, a_samples)
      output[result.T[0] == 0] = a_result
    else:
      output[result.T[0] == 0] = result[result.T[0] == 0]

    if node.right:
      b_samples = samples[result.T[0] == 1]
      b_result = self.ClassifyRecur(node.right, b_samples)
      output[result.T[0] == 1] = b_result
    else:
      output[result.T[0] == 1] = result[result.T[0] == 1]
    return output
################################################################################
################################################################################
class SVM(object):
  def __init__(self, classa, classb):
    self.classa = classa
    self.classb = classb
  def Train():
    pass
  def Classify():
    pass
  def ReformatData():
    pass
################################################################################
################################################################################
class TwoDGaussian(object):
  def __init__(self, x, y, variance, sign):
    self.x = x
    self.y = y
    self.mean = np.matrix([[x,y]]).T
    self.cov = np.eye(2) * variance
    self.cov_inv = la.inv(self.cov)
    self.sign = sign
    if self.cov != ():
      self.normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * self.cov)))
    else:
      self.normalizer = 1.0 / (np.sqrt(2.0 * np.pi * (self.cov + 0.000000001)))
    
  def __call__(self,sample):
      return self.sign * float(self.normalizer * np.exp(- 0.5 * (sample - self.mean).T * self.cov_inv * (sample - self.mean)))
 ###############################################################################
################################################################################
class DatasetGenerator(object):
  def __init__(self, gaussians, samples):
    '''gaussians=list of 2dGaussians
       num_samples=the number of points to generate'''
    self.distances = []
    for i in xrange(samples.shape[1]):
      distance = 0
      sample = samples[:,i]
      for gaussian in gaussians:
        # Compute normalizing term
        distance += gaussian(sample)
      self.distances.append(distance)
  def __call__(self):
    return self.distances
################################################################################

class ClassiferTest(object):
  def __init__(self, classifier, training_set):
    self.classifier   = classifier
    self.train_data, self.train_truth = self.classifier.ReformatData(training_set[0], training_set[1])
    # self.test_data, self.test_truth   = self.classifier.ReformatData(testing_set[0], testing_set[1])
    self.train_truth_raw = training_set[1]
    # self.test_truth_raw = testing_set[1]

  def Training(self):
    self.classifier.Train(self.train_data, self.train_truth)
  def Testing(self):
    self.train_result = self.classifier.Classify(self.train_data)
    # self.test_result  = self.classifier.Classify(self.test_data)
  def Results(self):
    compare = self.train_result - self.train_truth_raw
    compare = compare != 0

    a_compare = compare[self.train_truth_raw == 0]
    b_compare = compare[self.train_truth_raw == 1]

    error_rate = float(np.sum(compare)) / float(compare.shape[0])
    a_miss_class = float(np.sum(a_compare)) / float(a_compare.shape[1])
    b_miss_class = float(np.sum(b_compare)) / float(b_compare.shape[1])

    print error_rate
    print a_miss_class
    print b_miss_class

def GraphResults(results):
  pass

def GenerateTable(results):
  pass

def main():
  ''' Test the classes for performance and corrrecness'''
  data = CSVInput(sys.argv[1], first_row_titles=False)
  #truth_training = CSVInput(sys.argv[2], first_row_titles=False)
  samples = np.matrix(data.data)
  truth = samples[:,-1]
  samples = samples[:,:-1]

  #sets = np.array(truth_training.data)
  #training_samples = samples[sets.T[0] == 0]
  #trainging_truth = truth[sets.T[0] == 0]
  #testing_samples = samples[sets.T[0] == 1]
  #testing_truth = truth[sets.T[0] == 1]
  gaussians = []
  num_gaussians = 6
  parent_variance = 0.05
  leftX,leftY = np.random.multivariate_normal([0,0.5],[[parent_variance,0],[0,parent_variance]],num_gaussians/2).T
  rightX,rightY = np.random.multivariate_normal([1,0.5],[[parent_variance,0],[0,parent_variance]],num_gaussians/2).T
  minx = np.min([leftX,rightX])
  maxx = np.max([leftX,rightX])
  miny = np.min([leftY,rightY])
  maxy = np.max([leftY,rightY])
  leftX = (leftX - minx)/(maxx-minx)
  rightX = (rightX - minx)/(maxx-minx)
  leftY = (leftY - miny)/(maxy-miny)
  rightY = (rightY - miny)/(maxy-miny)
  #plt.plot(leftX,leftY,'ro')
  #plt.plot(rightX,rightY,'bo')
  #plt.show()
  imagesize = 25
  imgX,imgY = np.meshgrid(np.linspace(0,1,imagesize),np.linspace(0,1,imagesize))
  imgX = np.reshape(imgX,[1,imagesize*imagesize])
  imgY = np.reshape(imgY,[1,imagesize*imagesize])
  img = np.asmatrix(np.concatenate((imgX,imgY)))
  

  for i in range(num_gaussians/2):
    gaussians.append(TwoDGaussian(leftX[i],leftY[i],0.1,1))
    gaussians.append(TwoDGaussian(rightX[i],rightY[i],0.1,-1))
  gridData = DatasetGenerator(gaussians,img)
  image = gridData()
  image = np.reshape(image,[imagesize,imagesize])
  
  plt.imshow(image,interpolation='none',cmap='Greys_r')
  plt.show()
  pdb.set_trace()
  # print samples, samples.shape
  # print truth, truth.shape

  regression = Regression(0, 1)
  classify_test = ClassiferTest(regression, (training_samples, trainging_truth))
  classify_test.Training()
  classify_test.Testing()
  classify_test.Results()

  # fisher = Fisher(0, 1)
  # classify_test = ClassiferTest(fisher, (samples, truth))
  # classify_test.Training()
  # classify_test.Testing()
  # classify_test.Results()

  # random = Random(0, 1)
  # classify_test = ClassiferTest(random, (samples, truth))
  # classify_test.Training()
  # classify_test.Testing()
  # classify_test.Results()
  decision_tree = DecisionTree(Regression(0,1), 0,1, max_depth=800)
  dec_samples, dec_truth = decision_tree.ReformatData(training_samples, trainging_truth)
  decision_tree.Train(dec_samples, dec_truth)
  dec_result = decision_tree.Classify(dec_samples)
  print float(np.sum(dec_result != dec_truth)) / float(dec_result.shape[0])
  print dec_result.shape

  dec_samples, dec_truth = decision_tree.ReformatData(testing_samples, testing_truth)
  dec_result = decision_tree.Classify(dec_samples)
  print float(np.sum(dec_result != dec_truth)) / float(dec_result.shape[0])
  print dec_result.shape



  # GraphResults(results)
  # GenerateTable(results)

if __name__ == '__main__':
  main()
