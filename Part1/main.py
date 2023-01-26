from numpy.lib.polynomial import poly
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets.tests.test_samples_generator import make_blobs

# import and variables for circle data
import sklearn.datasets as dt
import matplotlib
seed = 11
rand_state = 11

# ceneters : groups No
# random_state : change shape , where to start points
X, y = make_blobs(n_samples = 200, centers=2 , random_state=200)

# uncomment for circle data
# X, y  = dt.make_circles(noise=0.06,random_state=rand_state)

# could use other kernels :linear , rbf , poly
clf = svm.SVC(kernel = 'linear', C=1000)
clf.fit(X,y)

predictData = [[3,4] , [5,6] , [8,9]]
print(clf.predict(predictData))

# plotting configs
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap = plt.cm.Paired )

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)

YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,Z , colors = 'k', levels =[-1,0,1],
           alpha=0.5,
           linestyles = ['--','-','--'])

ax.scatter(clf.support_vectors_[:,0],
           clf.support_vectors_[:,1],s=50,
           linewidth=1,facecolors = 'none')

plt.show()
