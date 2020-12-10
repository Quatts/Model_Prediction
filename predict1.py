from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import DateOffset
import datetime as dt
import urllib.request, json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy 
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from tensorflow import keras
#import plotly.plotly as py
#import plotly.offline as pyoff
#import plotly.graph_objs as go

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#set_session(sess)


#tf.debugging.set_log_device_placement(True)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.experimental.list_physical_devices('GPU')



AMDopen = pd.read_csv('BATS_MSFT_60.csv', delimiter=",", usecols=['open'],dtype={"open":object}, skip_blank_lines=0)
AMDclose = pd.read_csv('BATS_MSFT_60.csv', delimiter=",", usecols=['close'],dtype={"close":object}, skip_blank_lines=0)
AMDdate = pd.read_csv('BATS_MSFT_60.csv', delimiter=",", usecols=['time'],dtype={"time":object}, skip_blank_lines=0)
print(AMDopen)
AMDopen = AMDopen.open.tolist()
AMDclose = AMDclose.close.tolist()
AMDdate = AMDdate.time.tolist()
#AMDmid = []

AMDopen = [ float(x) for x in AMDopen if str(x) != 'nan']
AMDclose = [ float(x) for x in AMDclose if str(x) != 'nan']
AMDdate = [ x for x in AMDdate if x != 'nan']

print(len(AMDopen))
print(len(AMDclose))
print(len(AMDdate))
inputs = len(AMDopen)

AMDmid = []
AMDmid1 = AMDmid

for i in range(len(AMDopen)):
    AMDmid.append((AMDopen[i] + AMDclose[i])/2)

scaler=MinMaxScaler(feature_range=(0,1))
AMDmid=scaler.fit_transform(np.array(AMDmid).reshape(-1,1))
print("shape")
print(AMDmid.shape)
print(AMDmid)

training_size = int(len(AMDmid)*.75)
test_size = len(AMDmid)-training_size
train_data,test_data=AMDmid[0:training_size,:],AMDmid[training_size:(len(AMDmid)),:1]
#predict_data = np.zeros()
training_size,test_size

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		#dataY.append(dataset[i + time_step, 0])
		dataY.append(i)
	return numpy.array(dataX), numpy.array(dataY)
		
	

time_step = 12
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
#train_data = create_dataset()

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
print("shapes")
#print(X_test.shape[0])
#print(X_test.shape[1])
#print(ytest.shape[0])
#print(ytest.shape[1])
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1] , 1)
print(X_train)
print(y_train)
print("lm,aoooooooooooooooooooooooooooo")

model = load_model('model.h5')
#model=Sequential()
##model.add(LSTM(100,return_sequences=True,input_shape=(100,1)))
#model.add(LSTM(128,return_sequences=True,activation='relu',input_shape=(128,1)))
#model.add(LSTM(128,return_sequences=True))
#model.add(LSTM(128,return_sequences=True))
#model.add(LSTM(100,return_sequences=True))
#model.add(LSTM(50,return_sequences=True))
#model.add(Dropout(.3))
##model.add(LSTM(50))
#model.add(Flatten())
#model.add(Dense(1))
#model.compile(loss='mean_squared_error',optimizer='adam', metrics=['mean_squared_error'])
##model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])


model.summary()

#model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=200,batch_size=64,verbose=1)


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
#future = []
#currentStep = train_predict[:,-1:,:] #last step from the previous prediction
#future_pred_count = 1440
#for i in range(future_pred_count):
#    currentStep = model.predict(currentStep) #get the next step
#    future.append(currentStep) #store the future steps    
#
##after processing a sequence, reset the states for safety
#model.reset_states()
test_predict=model.predict(X_test)
#for i in range(len(ytest)):
#    print(str(i)+ ": x " + str(X_test[i])+" y "+str(ytest[i])+"\n")
    #y is equal to the 12th index of the next set of xtest
#fpredict = model.predict(train_data)
print("test_data")
print(len(test_data))
print("testpredict")
print(len(test_predict))
print("testpredict index")
print(scaler.inverse_transform(test_predict))
print("X_test")
print(len(X_test))
print("ytest")
print(len(ytest))
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
#AMDmid1=scaler.inverse_transform(AMDmid)
f= open("guru99.txt","a")
print(len(test_predict))
print(len(AMDmid1))
i = 0
while(i<817):
    f.write(str(i)+ ": x " + str(test_predict[i])+"\n")
    i+=1
#fpredict=scaler.inverse_transform(fpredict)
#print(fpredict)
print("hello")
mad = []
mad = AMDmid[len(train_predict):len(AMDmid)]

#for i in range(len(ytest)):
#    print(str(i)+ ": x " + str(test_predict[i])+" actual: "+str(mad[i])+"\n")



print(train_predict)
print("this guy right here")
print(test_predict)
print("this guy right her5675675675e")
print(len(test_predict))

print(len(mad))
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


look_back=12
trainPredictPlot = numpy.empty_like(AMDmid)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(AMDmid)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[(len(train_predict))+(look_back*2)+1:len(AMDmid)-1, :] = test_predict

#testPredictPlot[len(AMDmid)-12:len(AMDmid), :] = test_predict
#testPredictPlot1 = numpy.empty_like(AMDmid)
#testPredictPlot1[:, :] = numpy.nan
#testPredictPlot1[(len(train_predict))+(look_back*2)+1:, :] = test_predict
print("tp")
print(testPredictPlot)
print("tp math")
print((len(train_predict))+(look_back*2)+1)
print("amdmid")
print(len(AMDmid))
#the futurePlot is equal to the last "time step" of values genreated by the prediction method
#the future vales have already been predicted just clla them and plot them from... too tired to understand 
#bug tyler he'll understand, but figure out which len() is longer and that has the future values of "y's"
#futurePlot[:] = 



# plot baseline and predictions
plt.plot(scaler.inverse_transform(AMDmid[len(AMDmid)-200:]))
#plt.plot(AMDmid[len(AMDmid)-12:len(AMDmid)],"o")
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot[len(testPredictPlot)-200:])#len(testPredictPlot)])
#plt.plot(testPredictPlot1)
#plt.plot(currentStep)
plt.show()