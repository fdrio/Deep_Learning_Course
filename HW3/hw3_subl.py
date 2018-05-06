from sklearn import svm
import numpy as np
X = [[1, 2], [4, 2], [3, 5], [5, 1], [7, 2], [0, 6], [0, 0]]
X = np.array(X)
print(X)
y = [-1, -1, -1, 1, 1, -1, -1]

print(y)
print(X.shape)
clf = svm.SVC(kernel='linear', C=1E10)
clf.fit(X, y)
clf.coef_


import scipy.io
mat = scipy.io.loadmat("anascodata.mat")
x = np.array(np.matrix(mat["AnCultivo1Tr"]).T)


print(x.shape)


input_vec_training = []
input_vec_testing = []
num_of_samples_tr = []
num_of_samples_te = []
l = 0
for key in mat:
    if (l > 2):
        x = mat[key]

        print(np.array(x))
        print(type(x[0]))
        if (key.endswith("Tr")):
            for i in x:
                input_vec_training.append(i)
                num_of_samples_tr.append(x.shape[0])
        else:
            for i in x:
                input_vec_testing.append(i)
                num_of_samples_te.append(x.shape)
    l += 1
print(np.array(input_vec_training))
#print("Input testing vector ",input_vec_testing)
