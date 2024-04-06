
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
# Library to work with Regular Expressions
import re
#Split
from sklearn.model_selection import train_test_split
# This function makes the plot directly on browser
%matplotlib inline

# Seting a universal figure size
rcParams['figure.figsize'] = 10,8
#This librarys is to work with matrices
import pandas as pd
# This librarys is to work with vectors
import numpy as np
# This library is to create some graphics algorithmn
import seaborn as sns
# to render the graphs
import matplotlib.pyplot as plt
# import module to set some ploting parameters
from matplotlib import rcParams
# Library to work with Regular Expressions
import re
#Split
from sklearn.model_selection import train_test_split
# This function makes the plot directly on browser
%matplotlib inline

# Seting a universal figure size
rcParams['figure.figsize'] = 10,8
df = pd.read_csv('/content/Data (1).txt',delimiter='\t')
df.describe()
X = df.drop('Classification', axis=1)
Y = df.Classification
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)
print("This input data is consisted of",X.shape)
print("This dataset is consisted of",df.shape)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import graphviz
# Creating the model
model = Sequential()

# Inputing the first layer with input dimensions
model.add(Dense(18,
                activation='relu',
                input_dim=9,
                kernel_initializer='uniform'))
#The argument being passed to each Dense layer (18) is the number of hidden units of the layer.
# A hidden unit is a dimension in the representation space of the layer.

#Stacks of Dense layers with relu activations can solve a wide range of problems
#(including sentiment classification), and you’ll likely use them frequently.

# Adding an Dropout layer to previne from overfitting
model.add(Dropout(0.50))

#adding second hidden layer
model.add(Dense(32,
                kernel_initializer='uniform',
                activation='relu'))
# Adding an Dropout layer to previne from overfitting
model.add(Dropout(0.50))

#adding second hidden layer
model.add(Dense(32,
                kernel_initializer='uniform',
                activation='relu'))

# adding the output layer that is binary [0,1]
model.add(Dense(1,
                kernel_initializer='uniform',
                activation='sigmoid'))
#With such a scalar sigmoid output on a binary classification problem, the loss
#function you should use is binary_crossentropy

#Visualizing the model
model.summary()
#Creating an Stochastic Gradient Descent
sgd = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model
model.compile(optimizer = RMSprop(),
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
#optimizers list
#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# Fitting the ANN to the Training set
model.fit(X_train, y_train,
               batch_size = 30,
               epochs = 30, verbose=2)
scores = model.evaluate(X_train, y_train, batch_size=30)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
history = model.fit(X_train, y_train, validation_split=0.20,
                    epochs=180, batch_size=10, verbose=0)

# list all data in history
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()