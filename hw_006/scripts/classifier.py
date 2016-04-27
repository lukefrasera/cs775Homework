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
################################################################################
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

################################################################################
# SVM Utilities
################################################################################

def TestKKTConditions(alpha, sample, truth):
  return True
def GaussianKernel(query, sample):
  pass
################################################################################
################################################################################
class SVM(object):
  def __init__(self, classa, classb, C):
    self.classa = classa
    self.classb = classb
    #alphas is an ndarray
    self.alphas = np.zeros(1)
    self.support_vectors
    #the maximum weight
    self.C = C
  def Evaluate(self, sample, truth_class):
    """truth_class is an ndarray"""
    #result = np.sum(self.alphas*truth_class*GaussianKernel(sample,
  def OptimizePoint(self, alpha1, alpha2, sample1, sample2, truth1, truth2):
    if alpha1 < C or alpha1 > 0:
      #select a second sample
  def Train(self, samples, truth):
    #initialize alpha
    alphas = np.zeros(samples.shape[0])
    #loop through all the alpha weights
    num_alphas_changed = 0
    inspect_all = True
    while num_alphas_changed > 0 and inspect_all:
      num_alphas_changed = 0
      #loop over all the points which don't satisfy the Karush-Kahn-Tucker conditions
      for index,alpha in enumerate(alphas):
        #if inspect_all==True this is either the first iteration or we are verifying that we are done, so process every alpha
        #else this is not the first iteration and at least one alpha was changed on the last iteration
        if inspect_all :#or not TestKKTConditions(alpha, samples[index,:],truth[index]):
          #must check the weight of pair to ensure it is not 0 or C
          pair_index = random.random() * samples.shape[0]
          if pair_index == 0:
            pair_index += 1
          num_alphas_changed += self.OptimizePoint(alpha,alphas[pair_index],samples[index],samples[pair_index],truth[index],truth[pair_index])
      if inspect_all:
        inspect_all = False
      elif num_alphas_changed == 0:
        inspect_all = True
  def Classify(self, samples):
    pass
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
   
################################################################################
################################################################################

def main():
  ''' Test the classes for performance and corrrecness'''
  #Generate a number of points in 2D, these will be the gaussian centers
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
  gridData.plot_grid()
  #generate the actual dataset to classify on
  num_training_samples = 1000
  samples = np.random.rand(num_training_samples,2)
  sampleData = DatasetGenerator(gaussians,samples)
  sampleData.plot_sparse()
  #perform the classification
  
  plt.show()
  
if __name__ == '__main__':
  main()
