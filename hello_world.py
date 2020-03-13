from sklearn import tree

############################# Training Data #############################
# Used to train a classifier, more examples is the better
############################### Test Data ###############################
# Data which was not provided as training and is used to test the classifier
#########################################################################

# Features === input
# [weight, texture]
# 1 - smooth, 0 - bumpy
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# Labels === output
# 1 - orange, 0 - apple
labels = [0, 0, 1, 1]

##########################################################################

# A classifier is like a 'box of rules' which will decide what the input data is, based on
# the provided training data
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

# Receive the features of the new object as input
print(classifier.predict([[140, 1], [150, 0]]))
