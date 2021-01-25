import os
import pandas as pd 
import numpy as np
import glob

heart_rate_data = pd.DataFrame()
for heart_rate_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/heart_rate-20*')):       
   df = pd.read_json(heart_rate_file)
   bpm = list()
   index_list = list(range(len(df.index)))
   for index in index_list:
       bpm_point = df.loc[index, 'value']['bpm']
       bpm_point = np.float64(bpm_point)
       bpm.append(bpm_point)
   bpm = pd.Series(bpm)
   da = pd.concat([df['dateTime'], bpm], axis = 1)
   heart_rate_data = heart_rate_data.append(da) 

sleep_data = pd.DataFrame() 
for sleep_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Sleep/sleep-20*')):
   df = pd.read_json(sleep_file)
   da = pd.concat([pd.to_datetime(df['dateOfSleep']), df['minutesAsleep']], axis = 1)
   sleep_data = sleep_data.append(da)

exercise_data = pd.DataFrame() 
for exercise_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/exercise*')):
   df = pd.read_json(exercise_file, dtype = True)
   da = pd.DataFrame([df['startTime'], df['activityName'], df['duration']*1.6667e-5]) #convert duration from milliseconds to min
   da = da.transpose()
   da['startTime'] = pd.to_datetime(da['startTime'])
   exercise_data = exercise_data.append(da) 

print(heart_rate_data.head())
print(sleep_data.head())
print(exercise_data.head())

start = datetime.time(1,0,0)
end = datetime.time(5,30,0)

heart_rate_data.columns = ['dateTime', 'bpm']
heart_rate_data['dateTime'] = pd.DatetimeIndex(heart_rate_data['dateTime'])
heart_rate_data.set_index(keys = 'dateTime', inplace = True)
rhrData = heart_rate_data.between_time(start, end) #Return DataFrame with data between 1 AM and 5:30 AM, average of which will be RHR
rhrData.reset_index(inplace = True)
rhrData.columns = ['dateTime', 'bpm']
rhrData['date'] = rhrData.loc[:,'dateTime'].dt.date
rhr_df = pd.pivot_table(rhrData, values = 'bpm', index = 'date', aggfunc = np.mean)

exercise_data['date'] = exercise_data['startTime'].dt.date
exercise_data_df = pd.pivot_table(exercise_data, values = 'duration', columns = 'activityName', index = 'date', aggfunc = np.sum)
exercise_data_df.drop(['Run', 'Outdoor Bike'], axis = 1, inplace = True)   #Run and outdoor bike activities are completely null
exercise_data_df = exercise_data_df.fillna(0)
exercise_data_df['workout'] = exercise_data_df['Aerobic Workout'] + exercise_data_df['Sport']  #Working out can register on Fitbit as either of these
exercise_data_df.drop(['Aerobic Workout', 'Sport'], axis = 1, inplace = True) #Don't need this now, just clean up step

sleep_data.set_index('dateOfSleep', inplace = True)

all_data = exercise_data_df.join([sleep_data, rhr_df])
all_data.reset_index(inplace = True)
all_data.drop('index', axis = 1, inplace = True)
index_list = list(range(0, round(len(all_data))))
np.random.shuffle(index_list)
trainIndex = index_list[0:(round(0.8*len(index_list)))]
crossValIndex = index_list[len(trainIndex): round((((len(index_list)-len(trainIndex))/2) + len(trainIndex)))]
testIndex = index_list[(len(train_index) + len(crossValIndex)) :]

X = all_data.iloc[:,:-1]
Xtrain = X.loc[trainIndex, :]
Xtrain = Xtrain.to_numpy()
Xtrain = np.c_[np.ones((Xtrain.shape[0],1)), Xtrain]
y = all_data.loc[:,'bpm'].to_numpy()
ytrain = y[trainIndex]

XcrossVal = X.loc[crossValIndex, :].to_numpy()
ycrossVal = y[crossValIndex]
test = X.loc[testIndex, :].to_numpy()
ytest = y[testIndex]

theta_best = np.linalg.inv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(ytrain) #use normal equation to find theta
y_predict = XcrossVal * theta_best