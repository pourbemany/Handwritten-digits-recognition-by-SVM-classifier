import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import copy

def convert_to_csv(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
        
    # df = pd.DataFrame()    
    for image in images:
        list = [str(pix) for pix in image]
        # df.loc[len(df)] = list
        # df = df.append(pd.DataFrame([list]))
        # df = df.append(pd.Series(list), ignore_index=True)
        o.write(",".join(list)+"\n")
    f.close()
    o.close()
    l.close()

def svm_classifing(X_train,y_train,X_test,y_test,kernel_type):
    start_time = time.time()
    if kernel_type == 'poly':
        svclassifier = SVC(kernel=kernel_type, degree=8)
    else:
        svclassifier = SVC(kernel=kernel_type)                
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    print("*********************************************************************************")
    print("Traning time for %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_train)), (time.time() - start_time)))
    print("*********************************************************************************")
    print("score = %3.2f" %svclassifier.score(X_test,y_test))
    
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    return y_pred

def show_samples(X_test, y_pred, rand_list, name):
    for i in rand_list:
        two_d = (np.reshape(X_test.values[i], (28, 28))).astype(np.uint8)
        plt.title('predicted label: {0}'. format(y_pred[i]))
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        plt.savefig('fig/'+name+'_'+str(i)+'.jpg')
        plt.show()

#*******************************main******************************
start_time = time.time()     
convert_to_csv("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_handwritten.csv", 22000)

df = pd.read_csv('mnist_handwritten.csv', sep=',', header=None)    
print("converting time: --- %s seconds ---" % (time.time() - start_time))
print(df.head(3))
print(df.shape)

sns.countplot(df[0])
plt.show()

  
# df_class = df.iloc[:,1:]
# df_data = df.iloc[:, 0]

#(train set: image indexes 0-1000) and (test set: image indexes 2000-2100)
X_train = df.iloc[:1000,1:]
y_train = df.iloc[:1000, 0]

X_test = df.iloc[2000:2100,1:]
y_test = df.iloc[2000:2100, 0]

svm_classifing(X_train,y_train,X_test,y_test,'linear')

#(train set: image indexes 0-10000) and (test set: image indexes 20000-20100)
X_train = df.iloc[:10000,1:]
y_train = df.iloc[:10000, 0]

X_test = df.iloc[20000:20100,1:]
y_test = df.iloc[20000:20100, 0]

svm_classifing(X_train,y_train,X_test,y_test,'linear')

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df.iloc[:10000,1:]
y_train = df.iloc[:10000, 0]

X_test = df.iloc[20000:21000,1:]
y_test = df.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df.iloc[:10000,1:]
y_train = df.iloc[:10000, 0]

X_test = df.iloc[20000:21000,1:]
y_test = df.iloc[20000:21000, 0]

svm_classifing(X_train,y_train,X_test,y_test,'poly')

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df.iloc[:10000,1:]
y_train = df.iloc[:10000, 0]

X_test = df.iloc[20000:21000,1:]
y_test = df.iloc[20000:21000, 0]

svm_classifing(X_train,y_train,X_test,y_test,'rbf')

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df.iloc[:10000,1:]
y_train = df.iloc[:10000, 0]

X_test = df.iloc[20000:21000,1:]
y_test = df.iloc[20000:21000, 0]

svm_classifing(X_train,y_train,X_test,y_test,'sigmoid')

rand_list = (np.random.randint(0,1000,6))
show_samples(X_test, y_pred, rand_list, 'original')
    
#***************************convert images to binary threshold 200
# df1 = pd.DataFrame()      
# #df1[:]=np.where(df<100,0,1)   
df1 = copy.deepcopy(df)

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df1.iloc[:10000,1:]
X_train[X_train <= 200] = 0
X_train[X_train > 200] = 1
y_train = df1.iloc[:10000, 0]

X_test = df1.iloc[20000:21000,1:]
X_test[X_test <= 200] = 0
X_test[X_test > 200] = 1
y_test = df1.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

show_samples(X_test, y_pred, rand_list, 'binary_200')
   
#***************************convert images to binary threshold 150
df1 = copy.deepcopy(df)

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df1.iloc[:10000,1:]
X_train[X_train <= 150] = 0
X_train[X_train > 150] = 1
y_train = df1.iloc[:10000, 0]

X_test = df1.iloc[20000:21000,1:]
X_test[X_test <= 150] = 0
X_test[X_test > 150] = 1
y_test = df1.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

show_samples(X_test, y_pred, rand_list, 'binary_150')

#***************************convert images to binary threshold 100
df1 = copy.deepcopy(df)

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df1.iloc[:10000,1:]
X_train[X_train <= 100] = 0
X_train[X_train > 100] = 1
y_train = df1.iloc[:10000, 0]

X_test = df1.iloc[20000:21000,1:]
X_test[X_test <= 100] = 0
X_test[X_test > 100] = 1
y_test = df1.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

show_samples(X_test, y_pred, rand_list, 'binary_100')
    
#***************************convert images to binary threshold 50
df1 = copy.deepcopy(df)

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df1.iloc[:10000,1:]
X_train[X_train <= 50] = 0
X_train[X_train > 50] = 1
y_train = df1.iloc[:10000, 0]

X_test = df1.iloc[20000:21000,1:]
X_test[X_test <= 50] = 0
X_test[X_test > 50] = 1
y_test = df1.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

show_samples(X_test, y_pred, rand_list, 'binary_50')
    
#***************************convert images to binary threshold 1
df1 = copy.deepcopy(df)

#(train set: image indexes 0-10000) and (test set: image indexes 20000-21000)
X_train = df1.iloc[:10000,1:]
X_train[X_train <= 1] = 0
X_train[X_train > 1] = 1
y_train = df1.iloc[:10000, 0]

X_test = df1.iloc[20000:21000,1:]
X_test[X_test <= 1] = 0
X_test[X_test > 1] = 1
y_test = df1.iloc[20000:21000, 0]

y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')

show_samples(X_test, y_pred, rand_list, 'binary_1')

    
print("total time: --- %s seconds ---" % (time.time() - start_time))    