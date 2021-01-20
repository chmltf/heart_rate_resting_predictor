import os
import pandas as pd 

#heart_rate = pd.read_json(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/heart_rate-2020-12-19.json'))
#heart_rate_list = glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/heart_rate*'))

heart_rate_data = pd.DataFrame()
for heart_rate_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/heart_rate*')):       
   df = pd.read_json(heart_rate_file, dtype = True)
   heart_rate_data.append(df) #need to figure out how to get bpm out of value column

sleep_data = pd.DataFrame() #works in that we get sleep_data, but get traceback
for sleep_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Sleep/sleep*')):
   df = pd.read_json(sleep_file, dtype = True)
   da = pd.DataFrame([pd.to_datetime(df['dateOfSleep']), df['minutesAsleep']])
   da = da.transpose()
   sleep_data = sleep_data.append(da)

exercise_data = pd.DataFrame()  #This works!!
for exercise_file in glob.glob(os.path.join(os.getcwd(), 'LaurenFleming_original_data/Physical_Activity/exercise*')):
   df = pd.read_json(exercise_file, dtype = True)
   da = pd.DataFrame([df['startTime'], df['activityName'], df['duration']*1.6667e-5]) #convert duration from milliseconds to min
   da = da.transpose()
   da['startTime'] = pd.to_datetime(da['startTime'])
   exercise_data = exercise_data.append(da) 