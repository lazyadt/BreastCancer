import numpy as np
import csv
import os
from keras.models import Sequential
from keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'             # used because it additionally filters out errors

# Opening and reading the file
file = open(r"breast-cancer-wisconsin.csv","r")
dataDict = csv.DictReader(file,delimiter=",")

# Column names, if you do not want to print this, you can comment it
print(dataDict.fieldnames)

# Creating a 2D array by extracting the column id's 
data = []
for rowDict in dataDict:
    del rowDict["Id"]
    data.append( [ int(rowDict[v]) if rowDict[v].isdigit()else 0 for v in dataDict.fieldnames[1:] ] )

# Converting to numpy array
data  = np.asarray(data)
print(data.shape)

X = data[:,0:8]
Y = data[:,8]
print("Data:",X[0],"Result:",Y[0])

print("Creating Model")
model = Sequential()
model.add(layers.Dense(32,activation=layers.activations.relu,input_shape=(9,)))
model.add(layers.Dense(32,activation=layers.activations.relu))
model.add(layers.Dense(32,activation=layers.activations.softmax))

# Using adam optimizer beacuse it has (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Result 2 means benign stage, 4 means malignant stage for the tumor
def ReturnResult(value):
    if(value == 2):
        return "Benign stage"
    elif(value == 4):
        return "Malignant stage"
    else:
        return "Uncertain"

for q in range(5):
    print("Model training:",q)
    model.fit(X,Y,batch_size=512,epochs=10,validation_split=0.1,verbose=2,shuffle=True)

    print("...In progress...")
    for i in range(5):
        index = np.random.randint(600,len(X)-1)
        pred = model.predict_classes(X[index].reshape(1,9),verbose=0)

        if pred[0] == Y[index][0]:
            print("Answer:", ReturnResult(Y[index][0]),Y[index][0], "Result:", ReturnResult(pred[0]),pred[0])
        else:
            print("Answer:", ReturnResult(Y[index][0]),Y[index][0], "Result:", ReturnResult(pred[0]),pred[0])