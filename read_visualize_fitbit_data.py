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