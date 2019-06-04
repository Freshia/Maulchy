import csv
from sklearn import datasets,preprocessing,cross_validation,neighbors,model_selection
import numpy as np
import matplotlib.pyplot as plt
import cv2
trainFile=open('F:/Datasets/MNIST Sign Language/sign_mnist_train.csv')
testFile=open('F:/Datasets/MNIST Sign Language/sign_mnist_test.csv')
trainreader = csv.reader(trainFile)
#testreader = csv.reader(testFile)
for row in trainreader:
    trainrows = list(trainreader)
# for row in testreader:
#     testrows = list(testreader)

trainTargets=[]
trainEntries=[]
testTargets=[]
testEntries=[]

#Defining first row as targets and all the rest as the image pixels
for labels in trainrows:
    trainTargets.append(labels[0])
    trainEntries.append(labels[1:])
# for labels in testrows:
#     testTargets.append(labels[0])
#     testEntries.append(labels[1:])

x_train,x_test, y_train, y_test = cross_validation.train_test_split(trainEntries,trainTargets,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test,y_test)
print('ACCURACY')
print(accuracy)
false=0
true=0
# for i in range(len(testEntries)):
#     prediction = clf.predict(np.array(testEntries[i]).reshape(1,-1))
#     truth = testTargets[i]
#     if (prediction==truth):
#         true+=1
#     else:
#         false+=1
# myaccuracy = true/(true+false)
# print("My Accuracy")
# print(myaccuracy)

img1_arr = np.asarray(trainEntries[3],dtype=float)
img1_2d = np.reshape(img1_arr, (28, 28))
 # show it
plt.subplot(111)
plt.imshow(img1_2d, cmap=plt.get_cmap('gray'))
plt.show()


cam=cv2.VideoCapture(0)
cam.set(3,28)
cam.set(4,28)
while True:
    _,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow('webcamgray',gray)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

# free up memory
cam.release()
cv2.destroyAllWindows()




