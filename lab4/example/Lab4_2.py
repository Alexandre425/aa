# -*- coding: utf-8 -*-
"""

@author:    Manuel Diniz, 84125
            Alexandre Rodrigues, 90002
            
"""

from tensorflow.keras.datasets import fashion_mnist as fmds
from tensorflow import keras as keras
from tensorflow.keras import layers as ly
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy as sp



# Load data1
xtrain = np.load('data1_xtrain.npy')
ytrain = np.load('data1_ytrain.npy')
xtest = np.load('data1_xtest.npy')
ytest = np.load('data1_ytest.npy')


# Number of Classes and Probability of each class
size_ytrain = len(ytrain)
classes = []
classes_cnt = []
j = -1

for i in range(size_ytrain):
    if ytrain[i, 0] not in classes:
        classes.append(ytrain[i, 0])
        classes_cnt.append(1)
        j += 1
    else:
        classes_cnt[j] += 1
        
num_classes = len(classes)
classes_probs = [c / size_ytrain for c in classes_cnt]

    
#Plot Train Scatter
c1_train = xtrain[:50]
c2_train = xtrain[50:100]
c3_train = xtrain[100:]

plt.figure()
plt.scatter(c1_train[:, 0], c1_train[:, 1], label='Class1')
plt.scatter(c2_train[:, 0], c2_train[:, 1], label='Class2')
plt.scatter(c3_train[:, 0], c3_train[:, 1], label='Class3')
plt.xlim(-5,8)
plt.ylim(-5,8)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Train')
plt.legend()
plt.show()


#Plot Test Scatter
c1_test = xtest[:50]
c2_test = xtest[50:100]
c3_test = xtest[100:]

plt.figure()
plt.scatter(c1_test[:, 0], c1_test[:, 1], label='Class1')
plt.scatter(c2_test[:, 0], c2_test[:, 1], label='Class2')
plt.scatter(c3_test[:, 0], c3_test[:, 1], label='Class3')
plt.xlim(-5,8)
plt.ylim(-5,8)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Test')
plt.legend()
plt.show()



    ###################
    #   Naive Bayes   #
    ###################

    
    
# Mean
mean_c1 = np.array([np.mean(c1_train[:,0]), np.mean(c1_train[:,1])])
mean_c2 = np.array([np.mean(c2_train[:,0]), np.mean(c2_train[:,1])])
mean_c3 = np.array([np.mean(c3_train[:,0]), np.mean(c3_train[:,1])])


# Vídeo Árbitro (var)
diag_c1 = np.diag([np.var(c1_train[:,0]), np.var(c1_train[:,1])])
diag_c2 = np.diag([np.var(c2_train[:,0]), np.var(c2_train[:,1])])
diag_c3 = np.diag([np.var(c3_train[:,0]), np.var(c3_train[:,1])])


# Multivariate Gaussian Distribution of each class
# P(ci|x) = P(ci) * P(x|ci)
p_c1_x = classes_probs[0] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c1, diag_c1)
p_c2_x = classes_probs[1] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c2, diag_c2)
p_c3_x = classes_probs[2] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c3, diag_c3)


# Error Percentage
prediction = np.zeros([150, 1])
for i in range(0, 150):
    p_c_x = np.array([p_c1_x[i], p_c2_x[i], p_c3_x[i]])
    prediction[i,0] = np.argmax(p_c_x) + 1

print('Error percentage NaiveBayes: ', (1-accuracy_score(ytest, prediction))*100, '%')


# Plot Scatter Prediction
plt.figure()
plt.scatter(xtest[0:11, 0], xtest[0:11, 1], c='#1f77b4')
plt.scatter(xtest[11:13, 0], xtest[11:13, 1], c='#2ca02c')
plt.scatter(xtest[13:23, 0], xtest[13:23, 1], c='#1f77b4')
plt.scatter(xtest[23, 0], xtest[23, 1], c='#2ca02c')
plt.scatter(xtest[24:50, 0], xtest[24:50, 1], c='#1f77b4')
plt.scatter(xtest[50:55, 0], xtest[50:55, 1], c='#ff7f0e')
plt.scatter(xtest[55, 0], xtest[55, 1], c='#2ca02c')
plt.scatter(xtest[56:101, 0], xtest[56:101, 1], c='#ff7f0e')
plt.scatter(xtest[101:110, 0], xtest[101:110, 1], c='#2ca02c')
plt.scatter(xtest[110, 0], xtest[110, 1], c='#ff7f0e')
plt.scatter(xtest[111:132, 0], xtest[111:132, 1], c='#2ca02c')
plt.scatter(xtest[132, 0], xtest[132, 1], c='#ff7f0e')
plt.scatter(xtest[133:140, 0], xtest[133:140, 1], c='#2ca02c')
plt.scatter(xtest[140, 0], xtest[140, 1], c='#1f77b4')
plt.scatter(xtest[141:150, 0], xtest[141:150, 1], c='#2ca02c')
plt.xlim(-5,8)
plt.ylim(-5,8)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Test Classification Naive Bayes')
plt.legend()
plt.show()



    #############
    #   Bayes   #
    #############



# Covariance Matrix
cov_c1 = np.cov((c1_train[:,0]),(c1_train[:,1]))
cov_c2 = np.cov((c2_train[:,0]),(c2_train[:,1]))
cov_c3 = np.cov((c3_train[:,0]),(c3_train[:,1]))


# Multivariate Gaussian Distribution of each class
# P(ci|x) = P(ci) * P(x|ci)
p_c1_x = classes_probs[0] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c1, cov_c1)
p_c2_x = classes_probs[1] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c2, cov_c2)
p_c3_x = classes_probs[2] * sp.stats.multivariate_normal.pdf(xtest[:], mean_c3, cov_c3)


# Error Percentage
prediction = np.zeros([150, 1])
for i in range(0, 150):
    p_c_x = np.array([p_c1_x[i], p_c2_x[i], p_c3_x[i]])
    prediction[i,0] = np.argmax(p_c_x) + 1 

print('Error Percentage Bayes: ', (1-accuracy_score(ytest, prediction))*100 , '%')


# Plot Scatter Prediction
plt.scatter(xtest[0:50, 0], xtest[0:50, 1], c='#1f77b4')
plt.scatter(xtest[50:55, 0], xtest[50:55, 1], c='#ff7f0e')
plt.scatter(xtest[55, 0], xtest[55, 1], c='#2ca02c')
plt.scatter(xtest[56:100, 0], xtest[56:100, 1], c='#ff7f0e')
plt.scatter(xtest[100:110, 0], xtest[100:110, 1], c='#2ca02c')
plt.scatter(xtest[110, 0], xtest[110, 1], c='#ff7f0e')
plt.scatter(xtest[111:132, 0], xtest[111:132, 1], c='#2ca02c')
plt.scatter(xtest[132, 0], xtest[132, 1], c='#ff7f0e')
plt.scatter(xtest[133:140, 0], xtest[133:140, 1], c='#2ca02c')
plt.scatter(xtest[140, 0], xtest[140, 1], c='#ff7f0e')
plt.scatter(xtest[141:150, 0], xtest[141:150, 1], c='#2ca02c')
plt.xlim(-5,8)
plt.ylim(-5,8)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Test Classification Bayes')
plt.legend()
plt.show()