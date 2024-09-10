import pandas as pd

######################## PART 1 #####################################################
def getTrainingData(insulin_csv, cgm_csv):
    insulin_columns_to_read = ['Date', 'Time', 'BWZ Carb Input (grams)']
    df_insulin = pd.read_csv(insulin_csv, usecols=insulin_columns_to_read)
    df_insulin['Date'] = pd.to_datetime(df_insulin['Date'], errors='coerce').dt.date
    df_insulin['Date'] = df_insulin['Date'].astype(str)
    df_insulin['datetime'] = pd.to_datetime(df_insulin['Date'] + ' ' + df_insulin['Time'], errors='coerce')
    df_insulin = df_insulin.drop(columns=['Date', 'Time'])
    df_insulin = df_insulin.rename(columns={'BWZ Carb Input (grams)': 'meal'})
    df_insulin = df_insulin.dropna(subset=['meal'])
    df_insulin = df_insulin[df_insulin['meal'] != 0]
    df_insulin = df_insulin.sort_values(by='datetime').reset_index(drop=True)

    del insulin_columns_to_read

    meal_time = []
    # Loop through the DataFrame to find meal times with a > 2-hour gap
    for i in range(len(df_insulin) - 1):
        current_time = df_insulin['datetime'].iloc[i]
        next_time = df_insulin['datetime'].iloc[i + 1]

        if current_time + pd.Timedelta(hours=2) < next_time:
            meal_time.append(current_time)

    cgm_columns_to_read = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
    df_cgm = pd.read_csv(cgm_csv, usecols=cgm_columns_to_read)
    df_cgm = df_cgm.rename(columns={'Sensor Glucose (mg/dL)': 'glucose'})
    df_cgm['datetime'] = pd.to_datetime(df_cgm['Date'] + ' ' + df_cgm['Time'])
    df_cgm = df_cgm.drop(columns=['Date', 'Time'])
    df_cgm = df_cgm.sort_values(by='datetime').reset_index(drop=True)

    del cgm_columns_to_read

    # find datetime for Sugar Glucose Level based on meal time, ex(meal time = 9:00, SugarGlucose measured after meal = 9:03), make meal time = 9:03
    index = 0
    new_meal_time = []

    for i in range(len(df_cgm)):
        if (df_cgm['datetime'].iloc[i] > meal_time[index]):
            if (df_cgm['datetime'].iloc[i] < (meal_time[index] + pd.Timedelta(minutes=5))): # Edge case: meal time and glucose measure not with in 5 min
                new_meal_time.append(df_cgm['datetime'].iloc[i])

            if (index < len(meal_time) - 1):
                index += 1

    meal_time = new_meal_time

    del new_meal_time
    del index

    # locate meal_time and add glucose level 30 min before and 2hr after meal_time, 30 data per row including meal_time glucose level
    total_glucose_data = 30 # 2hrs:30min / 5min
    num_glucose_before_meal = 6 # 30min / 5min

    meal_data = []
    no_meal_time = []

    for datetime in  meal_time:
        data = []
        index = df_cgm[df_cgm['datetime'] == datetime].index[0] - num_glucose_before_meal

        for row in range(index, index + total_glucose_data + 1):
            if (row < index + total_glucose_data):
                data.append(df_cgm.loc[row, 'glucose'])
            else:
                no_meal_time.append(df_cgm.loc[row, 'datetime'])

        meal_data.append(data)

    del index
    del data

    column_names = [f'col{i+1}' for i in range(len(meal_data[0]))]
    meal_data = pd.DataFrame(meal_data, columns=column_names)
    #-----------------------------Meal data (ready)------------------------------------------------------

    # --------------------------- start no meal data ---------------------------------------------------
    no_meal_data = []
    new_no_meal_time = []

    # adding start time of post_absorptive period given there is no next meal inbetween two hours of post aborptive period
    for i in range(len(no_meal_time) - 1):
        post_absorptive_period_start = no_meal_time[i]
        post_absorptive_period_end = post_absorptive_period_start + pd.Timedelta(hours=2)

        if (post_absorptive_period_start <= meal_time[i+1] and meal_time[i+1] < post_absorptive_period_end):
            continue

        new_no_meal_time.append(post_absorptive_period_start)

    no_meal_time = new_no_meal_time

    del new_no_meal_time

    for datetime in  no_meal_time:
        data = []
        try:
            index = df_cgm[df_cgm['datetime'] == datetime].index[0]
        except:
            print(datetime, 'datetime not found')

        for row in range(index, index + (total_glucose_data - num_glucose_before_meal)):
            data.append(df_cgm.loc[row, 'glucose'])

        no_meal_data.append(data)

    column_names = [f'col{i+1}' for i in range(len(no_meal_data[0]))]
    no_meal_data = pd.DataFrame(no_meal_data, columns=column_names)

    return meal_data, no_meal_data

meal_data, no_meal_data = getTrainingData('InsulinData.csv', 'CGMData.csv')
meal_data_1, no_meal_data_1 = getTrainingData('Insulin_patient2.csv', 'CGM_patient2.csv')
print(meal_data_1.head(5))
    ######################## PART 2 #####################################################
