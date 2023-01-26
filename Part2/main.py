# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# n_sample no pics
# n_feturea no pixels
# flatten the images to 1D array
n_samples = len(digits.images)
print(f'number of images : {n_samples}')
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
# gamma default = 0.001
# use different kernels
clf = svm.SVC(kernel = 'rbf', gamma=0.001 ,C = 1000)

# Split data into train and test subsets
# test_size --> 20%
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# check the correctness
print(f'Accurarcy : {accuracy_score(y_test , predicted)}')

# compelete report
print(
f"Classification report for classifier {clf}:\n"
f"{metrics.classification_report(y_test, predicted)}\n"
)
