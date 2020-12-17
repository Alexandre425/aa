import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from six import StringIO  
from IPython.display import Image  
import pydotplus



###########
# Dataset #
###########

xtest = np.load("Cancer_Xtest.npy")
xtrain = np.load("Cancer_Xtrain.npy")   
ytest = np.load("Cancer_Ytest.npy") 
ytrain = np.load("Cancer_Ytrain.npy")



##############
# SVM Linear #
##############

# Create a Linear SVM Classifier
clf = svm.SVC(kernel='linear')

# Train the model
clf.fit(xtrain, ytrain)

# Predict
y_pred_lin = clf.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred_lin))

# F-Measure
print('F-Measure: ', metrics.f1_score(ytest, y_pred_lin))

# Confusion Matrix
print('Confusion Matrix: ', metrics.confusion_matrix(ytest, y_pred_lin))



##################
# SVM Polynomial #
##################

# Create a Polynomial SVM Classifier
clf = svm.SVC(kernel='poly') # Linear Kernel

#Train the model
clf.fit(xtrain, ytrain)

# Predict
y_pred_pol = clf.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred_pol))

# F-Measure
print('F-Measure: ', metrics.f1_score(ytest, y_pred_pol))

# Confusion Matrix
print('Confusion Matrix: ', metrics.confusion_matrix(ytest, y_pred_pol))



###############
# Naive Bayes #
###############

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model
model.fit(xtrain,ytrain)

# Predict
y_pred = model.predict(xtest)

# Accuracy
print("Accuracy:", metrics.accuracy_score(ytest, y_pred))

# F-Measure
print('F-Measure: ', metrics.f1_score(ytest, y_pred))

# Confusion Matrix
print('Confusion Matrix: ', metrics.confusion_matrix(ytest, y_pred))



#################
# Decision Tree #
#################

# Create Decision Tree classifer
clf = DecisionTreeClassifier()

# Train Decision Tree
clf = clf.fit(xtrain,ytrain)

# Predict
y_pred = clf.predict(xtest)

# Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

# F-Measure
print('F-Measure: ',metrics.f1_score(ytest, y_pred))

# Confusion Matrix
print('Confusion Matrix: ',metrics.confusion_matrix(ytest,y_pred))

# Print Decision Tree do png
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('cancro.png')
Image(graph.create_png())

# Plot Decision Tree
tree.plot_tree(clf)