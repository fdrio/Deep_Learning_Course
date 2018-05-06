
# coding: utf-8

# In[2]:


from sklearn import svm
import numpy as np


# In[3]:


X = [[1, 2], [4, 2], [3, 5], [5, 1], [7, 2], [0, 6], [0, 0]]
X = np.array(X)
print(X)
y = [-1, -1, -1, 1, 1, -1, -1]

print(y)
print(X.shape)
clf = svm.SVC(kernel='linear', C=1E10)
clf.fit(X, y)
clf.coef_


# In[4]:


# magnitude of the parameters
# print("W vector: " clf.coef_)
print("Size of Beta Vector", np.linalg.norm(clf.coef_))
print("Size of the Margin", 2.0 / np.linalg.norm(clf.coef_))
#print("Parameters: ", clf.get_params)
alphas = np.abs(clf.dual_coef_)
print("Alphas: ", alphas)

support_vectors = clf.support_vectors_
print("Support vectors", support_vectors)
print("Equation intercept :", clf.intercept_)


# In[35]:


import scipy.io
mat = scipy.io.loadmat("anascodata.mat")
x = np.array(np.matrix(mat["AnCultivo1Tr"]).T)


print(x.shape)


input_vec_training = []
input_vec_testing = []
num_of_samples_tr = []
num_of_samples_te = []
l = 0

#Extracting test, training vectors and labels for each one

for key in mat:
    print(key)
    x = mat[key]
    if (l > 2):
        if(key.endswith("Tr")):
            x = np.array(list(zip(*x)))
            print(x.shape)
            for i in x:
                input_vec_training.append(list(i))
            num_of_samples_tr.append(x.shape[0])
        elif (key.endswith("Te")):
            x = np.array(list(zip(*x)))
            print(x.shape)
            for i in x:
                input_vec_testing.append(list(i))
            num_of_samples_te.append(x.shape[0])

    l += 1


# In[36]:


# creating labels vector
la = 0
labels_vec = []  # empty list containing the input vector
print(len(num_of_samples_tr))
for i in num_of_samples_tr:
    for x in range(i):
        labels_vec.append(la)
    la += 1
print(len(labels_vec))


# In[37]:


# Training the SVM
clf.fit(input_vec_training, labels_vec)


# In[60]:

# predicting the new label 
pred = clf.predict(input_vec_testing)
pred = list(pred)
print(len(pred))
for i in pred:
    print(i)


# In[71]:


count = 0
la = 0
labels_vec_te = []
for i in num_of_samples_te:
    for x in range(i):
        labels_vec_te.append(la)
    la += 1
print(labels_vec_te)


for i in range(len(labels_vec_te)):
    if (pred[i] != (labels_vec_te[i])):
        count += 1
print(count)
print(len(labels_vec_te))
