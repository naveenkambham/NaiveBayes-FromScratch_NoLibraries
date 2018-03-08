"""
Title:Implement Naive Bayes Classifier and run various test cases like using MLE, MAP estimates to understand the algorithm.
Developer : Naveen Kambham
Description: This is a simple Naive Bayes classifier implemented using MAP and MLE estimates for parameters. It has various test cases like
             using MLE estimates, MAP estimates and using the Leave One out cross validation to check the error rates.
"""

"""
Importing the required libraries. numpy - for nd arrays, math - for math operations
logsumpexp-for log sum tric, os - for file operations
"""


import numpy as np
import collections as col
from collections import defaultdict
import operator
from scipy.misc import logsumexp
import os
import math

""" ReadDataFromText(fileName) : [in] Filename [out] nparray
Method to read the data from text file.
"""

def ReadDataFromText(fileName):
    # Read the data in to np nd array
    dataArray = np.loadtxt(fileName)
    return dataArray


""" Predictor_MLE_Smoothing(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here using MLE estimates we are predicting a test set, smoothing is applied.
 Log function is used.
"""
def Predictor_MLE_Smoothing(testData,Priors,Likelihoods,SmoothingValues):
     correct =0

     # Creating an array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))

     # Iterating the test data.
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Theta value as in 3.5.2 and using alg 3.2
             Lic=0
             classification = key

             # Iterating the each column counter for Likelihoods
             for key,value in Likelihoods.items():

                 #Computing Theta
                 Theta= (((value[classification][row[key]]) +1) / (Priors[classification]+(SmoothingValues[key])))
                 Lic+= math.log(Theta)

             # using only Likelihood no prior since this is MLE. using Same terminology posterior to elimate the conflicts and make better understanding
             posterior = math.exp(Lic-logsumexp(priorsarray))
             #Adding the Class label and proabable value to the list
             ClassPosteriors.append([classification,posterior])

         # Sorting the list in asscending order by taking probable metric
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking against the true label with the predicted label and incrementing the correct counts
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct

""" Predictor_MLE_NoSmoothing(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here using MLE estimates we are predicting a test set, no smoothing is applied.
 Log function is used.
"""
def Predictor_MLE_NoSmoothing(testData,Priors,Likelihoods):
     correct =0

     # Creating array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Theta values as in 3.5.2 and using alg 3.2
             Lic=0
             classification = key

             # Iterating each column counter for likelihood counts
             for key,value in Likelihoods.items():
                 # Computing the theta
                 Theta= (((value[classification][row[key]])) / (Priors[classification]))
                 Lic+= np.log(Theta)
             posterior = math.exp(Lic-logsumexp(priorsarray))

             # Adding the classification and probability to list
             ClassPosteriors.append([classification,posterior])
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking for the correctness and incrementing the count
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct

""" Predictor_MLE_NoSmoothing_ConsiderPi(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here using MLE estimates we are predicting a test set, no smoothing is applied.
 Log function is used. Usually we should not consider prior for MLE but just for testing purpose i am adding this method.
"""
def Predictor_MLE_NoSmoothing_ConsiderPi(testData,Priors,Likelihoods,LenghtofTrainingData):
     correct =0

     # Creating array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Theta values as in 3.5.2 and using alg 3.2
             Pi = (value)/(LenghtofTrainingData)
             Lic = math.log(Pi)
             classification = key

             # Iterating each column counter for likelihood counts
             for key,value in Likelihoods.items():
                 # Computing the theta
                 Theta= (((value[classification][row[key]])) / (Priors[classification]))
                 Lic+= np.log(Theta)
             posterior = math.exp(Lic-logsumexp(priorsarray))

             # Adding the classification and probability to list
             ClassPosteriors.append([classification,posterior])
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking for the correctness and incrementing the count
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct

""" Predictor_MLE_NoSmoothing_Multiplication(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here MLE estimates are used and No smoothing. Since this is MLE we are not using the Priors
 Just for an understanding purpose Multiplication of probabilities is used. This is just a testing run created out of my curiousity.
 """
def Predictor_MLE_NoSmoothing_Multiplication(testData,Priors,Likelihoods):
     correct =0

     # Creating array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Theta values as in 3.5.2 and using alg 3.2 for prediction
             Theta=1
             classification = key

             # Iterating the each column counter for Likelihoods
             for key,value in Likelihoods.items():
                 Theta*= (((value[classification][row[key]])) / (Priors[classification]))

             # Appending the probability and classification values
             ClassPosteriors.append([classification,Theta])

         # Sorting the list in ascending order
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking whether predicted is correct or not.
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct


""" Predictor_MAP(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here Pi and Theta are caliculated using MAP estimates. Uniform priors
 are added.
"""
def Predictor_MAP_UniformPrior(testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData):
     correct =0

     # Creating array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Pi, Theta values as in 3.5.2 and using alg 3.2 and adding the uniform priors
             Pi = (value+1)/(LenghtofTrainingData+SmoothingValues[0])
             Lic = math.log(Pi)
             classification = key

             # Iterating column counters for likelihoods
             for key,value in Likelihoods.items():

                 # Computing Theta value and adding the normalization values
                 Theta= (((value[classification][row[key]]) +1) / (Priors[classification]+(SmoothingValues[key])))
                 Lic+= math.log(Theta)
             posterior = math.exp(Lic-logsumexp(priorsarray))
             # Appending the classification and probabilities to the list
             ClassPosteriors.append([classification,posterior])

         # Sorting the probabilities
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking whether the predicted label and actual label are correct or not and incrementing the no of correct parametersc
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct

""" Predictor_MAP(): [in] testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData [out]- No of correctly predicted testsets
 Method to predict test data based on the Priors and Likelihood counts. Here Pi and Theta are caliculated using MAP estimates. Uniform priors
 are added.
"""
def Predictor_MAP_2Prior(testData,Priors,Likelihoods,SmoothingValues,LenghtofTrainingData):
     correct =0

     # Creating array of prior values to use in LogExpSum method
     priorsarray= np.array(list(Priors.values()))
     for row in testData :
         # List to store the class labels and its predicted weight
         ClassPosteriors=[]

         #Iterating each Class label
         for key,value in Priors.most_common(len(Priors)):

             #Pi, Theta values as in 3.5.2 and using alg 3.2 and adding the uniform priors
             Pi = (value+50)/(LenghtofTrainingData+50*SmoothingValues[0])
             Lic = math.log(Pi)
             classification = key

             # Iterating column counters for likelihoods
             for key,value in Likelihoods.items():

                 # Computing Theta value and adding the normalization values
                 Theta= (((value[classification][row[key]]) +9) / (Priors[classification]+9*(SmoothingValues[key])))
                 Lic+= math.log(Theta)
             posterior = math.exp(Lic-logsumexp(priorsarray))
             # Appending the classification and probabilities to the list
             ClassPosteriors.append([classification,posterior])

         # Sorting the probabilities
         ClassPosteriors.sort(key=operator.itemgetter(1))

         # Checking whether the predicted label and actual label are correct or not and incrementing the no of correct parametersc
         if row[0] == int(ClassPosteriors[-1][0]):
            correct+=1
     return correct


"""
get_Priors_Likelihoods():[in] trainingdata [out] priors,Likelihoods,SmoothingValues.
Method to compute the priors and likelihoods by counting
"""
def get_Priors_Likelihoods(trainingData):
    #initilize the counters
     priors = col.Counter()
     #Using a dictionary to count observations in each corresponding column. So we will be using
     # different counters for different features x2,x3,...etc.
     Likelihoods={}

     # Getting the columns to initiliaze the counters at run time based on no of features
     (rows,columns) = trainingData.shape

     #initialize the default dict based on the feature values

    # This is to store the no of variants for each feature, suppose for column1 it will be 4 since we have values 0,1,2,3.
    # We will these counts for smoothing
     SmoothingValues={}
     for i in range(0,columns):

         # Max value - minimum value in a column plus one will give us the number of variants in a column
         SmoothingValues[i]= max(trainingData[:,i])-min(trainingData[:,i])+1

         #skipping the first column for likelihood counter initialziatio as we are using seperate counter for class labels to eliminate the ambiguity
         if i==0:
             continue

         #Initiliazing the default dictionary  for likelihoods at run time based on no of features
         Likelihoods[i]= defaultdict(col.Counter)


      # Counting the priors and likelihoods
     for row in trainingData:
         # Counting Priors
         priors[row[0]] += 1

         # Counting the likelihoods for features
         for key,value in Likelihoods.items():

             # Here Value is a default dictionary container with counter. So we are using class label at column 0 i.e row[0] and row[key] means
             # value at row[1],....row[7] i.e feature value. For example it will be combination of [0][0] or [0] [2] etc.
             value[row[0]][row[key]]+=1
     print(priors)
     print(Likelihoods)
     return (priors,Likelihoods,SmoothingValues)

def main():

  # Path of Data set files. Here assuming that test directory will have only data files with .txt format
  path= 'C:\Education\ML\AssignMent\Assignment2'

  # Creating list of .txt files
  datasets= [f for f in os.listdir(path) if f.endswith('.txt')]

  # Test Case - Prediction using MAP Estimates and Uniform prior
  print("Test Case - Prediction using MAP - Uniform Prior")
  for filepath in datasets :

     #Read the entire data in to a np array
     data= (ReadDataFromText(os.path.join(path, filepath)))

     # Running Leave One out Cross Validation
     correct =0

     # Iterating with c_index as flag so we get indexing based on the no of data rows.
     it = np.nditer(data, flags=['c_index'])

     while not it.finished:

         # Some times it.finished is not called because of which we may encounter buffer overflow error. So as a back up added this check so that even
         # in worst case we dont get this error.

         if it.index==len(data):
             break

         # Preparing the training data by leaving out current row
         traingSet = np.delete(data,it.index,0)

         #Get the priors and likelihoods for the trainng data
         (Priors,Likelihoods,UniformPriors)=get_Priors_Likelihoods(traingSet)

         #Predict the class lables. Sending the current row as test row
         correct += Predictor_MAP_UniformPrior(np.array([data[it.index]]),Priors,Likelihoods,UniformPriors,len(traingSet))
         it.iternext()
     print("For",filepath,"\n error rate:",(len(data)-correct)/len(data),"Accuracy :", correct/len(data) *100)


  # Test Case - Prediction using MAP Estimates and beta(2,2) Prior
  print("Test Case - Prediction using MAP - beta(2,2) Prior")
  for filepath in datasets :

     #Read the entire data in to a np array
     data= (ReadDataFromText(os.path.join(path, filepath)))

     # Running Leave One out Cross Validation
     correct =0

     # Iterating with c_index as flag so we get indexing based on the no of data rows.
     it = np.nditer(data, flags=['c_index'])

     while not it.finished:

         # Some times it.finished is not called because of which we may encounter buffer overflow error. So as a back up added this check so that even
         # in worst case we dont get this error.

         if it.index==len(data):
             break

         # Preparing the training data by leaving out current row
         traingSet = np.delete(data,it.index,0)

         #Get the priors and likelihoods for the trainng data
         (Priors,Likelihoods,UniformPriors)=get_Priors_Likelihoods(traingSet)

         #Predict the class lables. Sending the current row as test row
         correct += Predictor_MAP_2Prior(np.array([data[it.index]]),Priors,Likelihoods,UniformPriors,len(traingSet))
         it.iternext()
     print("For",filepath,"\n error rate:",(len(data)-correct)/len(data),"Accuracy :", correct/len(data) *100)

  # Test Case Prediction using MLE with smoothing. We can run this prediction in the above loop but to understand the things clear I am writing this seperately
  print("####################################### \n Test Case Prediction using MLE with Smoothing")
  for filepath in datasets :

     #Read the entire data in to a np array
     data= (ReadDataFromText(os.path.join(path, filepath)))
     # Running Leave One out Cross Validation
     correct =0
     it = np.nditer(data, flags=['c_index'])
     while not it.finished:
         if it.index==len(data):
             break
         # Preparing the training data by leaving out current row
         traingSet = np.delete(data,it.index,0)
         #Get the priors and likelihoods for the trainng data
         (Priors,Likelihoods,UniformPriors)=get_Priors_Likelihoods(traingSet)

         #Predict the class lables. Sending the current row as test row
         correct += Predictor_MLE_Smoothing(np.array([data[it.index]]),Priors,Likelihoods,UniformPriors)
         it.iternext()
     print("For",filepath,"\n error rate:",(len(data)-correct)/len(data),"Accuracy :", correct/len(data) *100)



  # Test Case Prediction using MLE with out smoothing. We can run this prediction in the above loop but to understand the things clear I am writing this seperately
  print("####################################### \n Test Case Prediction using MLE with out Smoothing")

  for filepath in datasets :

     #Read the entire data in to a np array
     data= (ReadDataFromText(os.path.join(path, filepath)))
     # Running Leave One out Cross Validation
     correct =0
     it = np.nditer(data, flags=['c_index'])
     while not it.finished:

         if it.index==len(data):
             break
         # Preparing the training data by leaving out current row
         traingSet = np.delete(data,it.index,0)
         #Get the priors and likelihoods for the trainng data
         (Priors,Likelihoods,UniformPriors)=get_Priors_Likelihoods(traingSet)

         #Predict the class lables. Sending the current row as test row
         correct += Predictor_MLE_NoSmoothing(np.array([data[it.index]]),Priors,Likelihoods)
         it.iternext()
     print("For",filepath,"\n error rate:",(len(data)-correct)/len(data),"Accuracy :", correct/len(data) *100)



# Test Case Prediction using MLE and considering Pi as well, with out Smoothing
  print("####################################### \n Test Case Prediction using MLE and considering Pi as well, with out Smoothing")

  for filepath in datasets :

     #Read the entire data in to a np array
     data= (ReadDataFromText(os.path.join(path, filepath)))
     # Running Leave One out Cross Validation
     correct =0
     it = np.nditer(data, flags=['c_index'])
     while not it.finished:
         if it.index==len(data):
             break
         # Preparing the training data by leaving out current row
         traingSet = np.delete(data,it.index,0)
         #Get the priors and likelihoods for the trainng data
         (Priors,Likelihoods,UniformPriors)=get_Priors_Likelihoods(traingSet)

         #Predict the class lables. Sending the current row as test row
         correct += Predictor_MLE_NoSmoothing_ConsiderPi(np.array([data[it.index]]),Priors,Likelihoods,len(traingSet))
         it.iternext()
     print("For",filepath,"\n error rate:",(len(data)-correct)/len(data),"Accuracy :", correct/len(data) *100)

main()
