#!/usr/bin/env python
import time
import numpy as np
from numpy import linalg as la
import numexpr as ne
import sys, csv
import pdb
import copy
import warnings
import random
import matplotlib.pyplot as plt
import pudb
################################################################################
################################################################################
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
                    value = -1
                  elif any(true in value for true in self.boolean_true):
                    value = 1
              row_list.append(value)
          self.data.append(row_list)
    self.rows = len(self.data)
    self.cols = len(self.data[0])

################################################################################
################################################################################
class Classifier(object):
  def __init__(self):
    pass
  def Train(self, samples, truth):
    pass
  def Classify(self, sample):
    pass
  def ReformatData(self, samples, truth):
    return (samples, truth)

################################################################################
# SVM Utilities
################################################################################

def MeetsKKTConditions(C, alpha, sample, truth, output):
  if alpha == 0 and (truth * output < 1):
    return False
  if 0 < alpha < C and (truth * output != 1):
    return False
  if alpha == C and (truth * output > 1):
    return False
  return True

def GaussianKernel(query, samples, cov):
  query = np.asmatrix(query)
  samples = np.asmatrix(samples)
  cov_inv = la.inv(cov)

  dist = cov_inv * (query - samples).T
  dist = np.multiply((query - samples).T,dist)
  dist = np.sum(dist,0).T
  normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * cov)))
  return normalizer * np.exp(-0.5 * dist)
################################################################################
################################################################################
class SVM(object):
  def __init__(self, C):
    #alphas is an ndarray
    self.alphas = np.zeros(1)
    #the maximum weight
    self.C = C
  def FindNonKKTSample(self, samples, truth, blacklist_index):
    num_samples = samples.shape[0]
    start_index = int(np.floor(random.random() * num_samples) - 1)
    for i in range(start_index,start_index+num_samples):
      sample_index = i % num_samples
      if sample_index == blacklist_index:
        continue
      sample = samples[sample_index]
      alpha = self.alphas[sample_index]
      sample_truth = truth[sample_index]
      output =  self.Evaluate(sample, samples, truth)
      if not MeetsKKTConditions(self.C, alpha,sample,sample_truth,output):
        return sample_index
    return -1
  def Evaluate(self, query, samples, truth_class):
    """truth_class is an ndarray"""
    variance = np.eye(3)
    the_sum = 0
    truth_class = truth_class.T[0]
    #pudb.set_trace()
    result = np.sign(self.alphas*truth_class*GaussianKernel(query,samples, variance))
    if result == 0:
      result = 1
    return result
    #return np.sign(self.alphas[i]*truth_class[i]*GaussianKernel(query,samples[i], variance))
  def EvaluateObjFunc(self, sample_index, samples, truth):
    alpha_sum = 0
    for alpha in self.alphas:
      alpha_sum += alpha

    right_sum = 0
    for ai,truthi,samplei in zip(self.alphas,truth,samples):   
      right_sum += np.sum(ai*np.asarray(self.alphas)*(truthi*np.asarray(truth)*np.asarray(GaussianKernel(samplei,samples, np.eye(3)))).T[0])
    return alpha_sum - 0.5 * right_sum
  def TakeStep(self, samples, truth, first_index, second_index):
    if first_index == second_index:
      raise ValueError("I thought I handled this case")
    alpha1 = self.alphas[first_index]
    alpha2 = self.alphas[second_index]
    truth1 = truth[first_index]
    truth2 = truth[second_index]
    sample1 = samples[first_index]
    sample2 = samples[second_index]
    error1 = self.Evaluate(sample1, samples, truth) - truth1
    error2 = self.Evaluate(sample2, samples, truth) - truth2
    s = truth1 * truth2
    if truth1 != truth2:
      L = max(0, alpha2 - alpha1)
      H = min(self.C, self.C + alpha2 - alpha1)
    else:
      L = max(0, alpha1 + alpha2 - self.C)
      H = min(self.C, alpha1 + alpha2)
    if L == H:
      return False
    covariance = np.eye(3)
    k11 = GaussianKernel(sample1,sample1, covariance)
    k12 = GaussianKernel(sample1,sample2, covariance)
    k22 = GaussianKernel(sample2,sample2, covariance)
    eta = 2 * k12 - k11 - k22
    if eta < 0:
      a2 = float(alpha2) - float(truth2) * float(error1-error2)/eta
      if a2 < L:
        a2 = L
      elif a2 > H:
        a2 = H
    else:
      temp_alphas = self.alphas
      self.alphas[second_index] = L
      Lobj = self.EvaluateObjFunc(second_index, samples, truth)
      self.alphas[second_index] = H
      Hobj = self.EvaluateObjFunc(second_index, samples, truth)
      self.alphas = temp_alphas
      if Lobj < Hobj + np.finfo(np.float).eps:
        a2 = L
      elif Lobj > Hobj + np.finfo(np.float).eps:
        a2 = H
      else:
        a2 = alpha2
    if a2 < 1e-8:
      a2 = 0
    elif a2 > self.C-1e-8:
      a2 = self.C
    if np.abs(a2-alpha2) < np.finfo(np.float).eps * (a2 + alpha2 + np.finfo(np.float).eps):
      return False
    a1 = alpha1 + s * (alpha2 - a2)
    self.alphas[first_index] = a1
    self.alphas[second_index] = a2
    return True
    # Update threshold to reflect change in Lagrange multipliers
    # Update weight vector to reflect change in a1 & a2, if SVM is linear
  def OptimizePoint(self, samples, truth, sample_index):
    #select a second sample which doesn't meet the KKT conditions
    second_index = self.FindNonKKTSample(samples, truth, sample_index)
    if second_index != -1:
      if self.TakeStep(samples, truth, second_index, sample_index):
        return 1
    for second_sample_index, alpha in enumerate(self.alphas):
      if second_sample_index != sample_index and alpha > 0 and alpha < self.C:
        if self.TakeStep(samples, truth, sample_index, second_sample_index):
          return 1
    num_samples = samples.shape[0]
    start_point = int(np.floor(random.random() * num_samples - 1))
    for i in range(start_point, start_point + num_samples):
      second_sample_index = i % num_samples
      if second_sample_index != sample_index and self.TakeStep(samples, truth, sample_index, second_sample_index):
        return 1
    return 0
  def Train(self, samples, truth):
    #initialize alpha
    self.alphas = np.zeros(samples.shape[0])
    self.truths = truth
    self.samples = samples
    #loop through all the alpha weights
    num_alphas_changed = 0
    inspect_all = True
    #iterations = 0
    while num_alphas_changed > 0 or inspect_all:
      #iterations += 1
      #if iterations % 1 == 0:
      num_alphas_changed = 0
      #loop over all the points which don't satisfy the Karush-Kahn-Tucker conditions
      for index,alpha in enumerate(self.alphas):

        #else this is not the first iteration and at least one alpha was changed on the last iteration
        sample = samples[index]
        sample_truth = truth[index]
        output = self.Evaluate(sample, samples, truth)
        if inspect_all or (alpha < self.C or alpha > 0):
          num_alphas_changed += self.OptimizePoint(samples, truth, index)
      if inspect_all:
        inspect_all = False
      elif num_alphas_changed == 0:
        inspect_all = True
  def Classify(self, samples):
    result = np.ndarray([samples.shape[0]])
    for i,sample in enumerate(samples):
      result[i] = self.Evaluate(sample, self.samples, self.truths)
    return result
  def ReformatData(self, samples, truth):
    pass
################################################################################
################################################################################
class TwoDGaussian(object):
  def __init__(self, x, y, variance, sign):
    self.x = x
    self.y = y
    self.mean = np.matrix([[x,y]])
    self.cov = np.eye(2) * variance
    self.cov_inv = la.inv(self.cov)
    self.sign = sign
    if self.cov != ():
      self.normalizer = 1.0 / (np.sqrt(la.det(2.0 * np.pi * self.cov)))
    else:
      self.normalizer = 1.0 / (np.sqrt(2.0 * np.pi * (self.cov + 0.000000001)))
    
  def __call__(self,sample):
      centeredSample = sample - self.mean
      dist = self.cov_inv * centeredSample.T
      dist = np.multiply(centeredSample.T,dist)
      dist = np.sum(dist,0).T
      return self.sign * self.normalizer * np.exp(- 0.5 * dist)
################################################################################
################################################################################
class DatasetGenerator(object):
  def __init__(self, gaussians, samples):
    '''gaussians=list of 2dGaussians
       num_samples=the number of points to generate'''
    self.samples = samples
    self.distances = np.zeros([samples.shape[0],1])
    for gaussian in gaussians:
      self.distances += gaussian(samples)
  def getDistances(self):
    return self.distances
  def plot_grid(self):
    #Get the image data and reformat it for matplotlib to graph
    image = self.distances
    image = np.reshape(image,[np.sqrt(image.shape[0]),np.sqrt(image.shape[0])])
    image[image > 0] = 1
    image[image < 0] = 0
    plt.figure(2)
    plt.imshow(image,interpolation='none',cmap='Greys_r')
    plt.show(block=False)
  def plot_sparse(self):
    #plot each point on a plane
    plt.figure(3)
    posClass = self.samples[(self.distances > 0).view(np.ndarray).ravel()==1]
    plt.plot(posClass[:,0],-posClass[:,1],'ro')
    negClass = self.samples[(self.distances <=0).view(np.ndarray).ravel()==1]
    plt.plot(negClass[:,0],-negClass[:,1],'bo')
    plt.show(block=False)
  def GetTruth(self):
    truth = self.distances
    truth[truth > 0] = 1
    truth[truth <= 0] = -1
    return truth
   
################################################################################
################################################################################

def main():
  ''' Test the classes for performance and corrrecness'''
  #Generate a number of points in 2D, these will be the gaussian centers
  np.random.seed(1)
  gaussians = []
  num_gaussians = 6
  parent_variance = 0.2
  leftX,leftY = np.random.multivariate_normal([0,0.5],[[parent_variance,0],[0,parent_variance]],num_gaussians/2).T
  rightX,rightY = np.random.multivariate_normal([1,0.5],[[parent_variance,0],[0,parent_variance]],num_gaussians/2).T
  
  #normalize the means so they are in [0,1] in both dimensions
  minx = np.min([leftX,rightX])
  maxx = np.max([leftX,rightX])
  miny = np.min([leftY,rightY])
  maxy = np.max([leftY,rightY])
  leftX = (leftX - minx)/(maxx-minx)
  rightX = (rightX - minx)/(maxx-minx)
  leftY = (leftY - miny)/(maxy-miny)
  rightY = (rightY - miny)/(maxy-miny)
  #create a meshgrid (i.e. a discrete gridding) of the  gaussian field so that we can visualize the boundary between the classes
  imagesize = 1000
  gridX,gridY = np.meshgrid(np.linspace(0,1,imagesize),np.linspace(0,1,imagesize))
  gridX = np.reshape(gridX,[1,imagesize*imagesize])
  gridY = np.reshape(gridY,[1,imagesize*imagesize])
  grid = np.asmatrix(np.concatenate((gridX,gridY))).T
  
  #build gaussians using the means previousl computed
  for i in range(num_gaussians/2):
    gaussians.append(TwoDGaussian(leftX[i],leftY[i],0.1,1))
    gaussians.append(TwoDGaussian(rightX[i],rightY[i],0.1,-1))
  #This function compute the weight of each point in meshgrid. For a discrete grid this is equivalent to building an image of the gaussian field
  gridData = DatasetGenerator(gaussians,grid)
  #gridData.plot_grid()
  #generate the actual dataset to classify on
  num_training_samples = 25
  train_samples = np.random.rand(num_training_samples,2)
  train_sample_data = DatasetGenerator(gaussians,train_samples)
  train_samples = np.concatenate((train_samples, np.ones((train_samples.shape[0], 1))), axis=1)

  num_test_samples = 100
  test_samples = np.random.rand(num_test_samples,2)
  test_sample_data = DatasetGenerator(gaussians,test_samples)
  #sampleData.plot_sparse()
  #perform the classification
  start = time.time()
  svm = SVM(100)
  svm.Train(train_samples, train_sample_data.GetTruth())
  train_classification = svm.Classify(train_samples)
  diff = train_classification - np.array(train_sample_data.GetTruth().T[0])
  num_errors = np.sum(diff!=0)
  print('Num errors: '+str(num_errors)+' '+str(float(num_errors)/num_training_samples))
  print time.time()-start
  plt.show()
  print svm.alphas
  
if __name__ == '__main__':
  main()
