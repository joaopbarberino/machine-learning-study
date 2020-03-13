import pydot
from sklearn.externals.six import StringIO
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
test_idx = [0, 50, 100]

# Training Data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing Data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

print(test_target)
print(classifier.predict(test_data))

dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data, feature_names=iris.feature_names,
                     class_names=iris.target_names, filled=True, rounded=True, impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')

