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
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
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
look_back = 15

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

train_generator = TimeseriesGenerator(AMDmid, AMDmid, length=look_back, batch_size=10)

#model = load_model('model.h5')
model=Sequential()
model.add(LSTM(10, activation="relu", input_shape=(look_back,1)))
model.add(Dense(1))
num_epochs = 30
model.compile(optimizer="adam", loss="mse")
model.fit(train_generator, epochs=num_epochs, verbose=1)
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
#train_predict=model.predict(X_train)
#future = []
#currentStep = train_predict[:,-1:,:] #last step from the previous prediction
#future_pred_count = 1440
#for i in range(future_pred_count):
#    currentStep = model.predict(currentStep) #get the next step
#    future.append(currentStep) #store the future steps    
#
##after processing a sequence, reset the states for safety
#model.reset_states()
#test_predict=model.predict(X_test)
#for i in range(len(ytest)):
#    print(str(i)+ ": x " + str(X_test[i])+" y "+str(ytest[i])+"\n")
    #y is equal to the 12th index of the next set of xtest
#fpredict = model.predict(train_data)
##Transformback to original form

#for i in range(len(ytest)):
#    print(str(i)+ ": x " + str(test_predict[i])+" actual: "+str(mad[i])+"\n")


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

# trainPredictPlot = numpy.empty_like(AMDmid)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(AMDmid)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[(len(train_predict))+(look_back*2)+1:len(AMDmid)-1, :] = test_predict

#testPredictPlot[len(AMDmid)-12:len(AMDmid), :] = test_predict
#testPredictPlot1 = numpy.empty_like(AMDmid)
#testPredictPlot1[:, :] = numpy.nan
#testPredictPlot1[(len(train_predict))+(look_back*2)+1:, :] = test_predict
# print("tp")
# print(testPredictPlot)
# print("tp math")
# print((len(train_predict))+(look_back*2)+1)
# print("amdmid")
# print(len(AMDmid))
#the futurePlot is equal to the last "time step" of values genreated by the prediction method
#the future vales have already been predicted just clla them and plot them from... too tired to understand 
#bug tyler he'll understand, but figure out which len() is longer and that has the future values of "y's"
#futurePlot[:] = 



# plot baseline and predictions
#plt.plot(scaler.inverse_transform(AMDmid[len(AMDmid)-200:]))
#plt.plot(AMDmid[len(AMDmid)-12:len(AMDmid)],"o")
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot[len(testPredictPlot)-200:])#len(testPredictPlot)])
#plt.plot(testPredictPlot1)
#plt.plot(currentStep)
#plt.show()

AMDmid.reshape((-1))

def predict(model, num_step):

	prediction_list = AMDmid[-look_back:]

	for _ in range(num_step):
		x = prediction_list[-look_back:]
		print(prediction_list[-look_back:])
		x = x.reshape(1, look_back, 1)
		out = model.predict(x)[0][0]
		prediction_list = np.append(prediction_list, out)
	
	prediction_list = prediction_list[look_back-1:]

	return prediction_list

def predict_dates(num_step):
	last_date = AMDdate[-1]
	prediction_dates = pd.date_range(last_date, periods=num_step+1).tolist()
	return prediction_dates
	
num_steps = 300
forecast = predict(model, num_steps)
forecast = forecast.reshape(-1,1)
forecast = scaler.inverse_transform(forecast)
prev_values = scaler.inverse_transform(AMDmid)
forecast = np.append(prev_values[:-look_back], forecast)

forecast_dates = predict_dates(num_steps)
forecast_dates = np.append(forecast_dates, AMDdate[:-num_steps])

prev_plot_data = np.array((prev_values[:-look_back]))
plot_data = np.array((forecast))


plt.plot(plot_data, label="Predicted")
plt.plot(prev_plot_data, label="Dataset")


plt.legend(loc="best")
plt.show()