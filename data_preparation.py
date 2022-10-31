#!/usr/bin/env python
# coding: utf-8

# In[150]:


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from EMD_lib import cubic_spline_3pts, EMD


# In[83]:


def get_data(df, mode='original', emd=False, dimension='one', window_size=0, intervals=2):
    delta_seconds = df.event_timestamp.diff() / np.timedelta64(1, 's')
    delta_seconds[np.where(delta_seconds == 0)[0]] = 1e-3
    delta_seconds = delta_seconds[1:]

    AP = df['Total_AP_energy_max']

    a = []
    for j, i in enumerate(df['Total_AP_energy_max']):
        try:
            i = float(i)
        except ValueError:
            a.append(j)
    for i in a:
        AP[i] = AP[i - 1]

    AP = np.diff(np.array(AP, dtype = float)) * 1000
    X = AP / delta_seconds
    df['total_AP'] = X
    if emd:
        ind = np.where(df.event_timestamp.apply(datetime.time) >= datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        
        #print(ind)
        prev = 0
        #df_help = pd.DataFrame(columns=df.columns)
        k = 0
        for i in ind[1:]:
            #print(i)
            #print(df.loc[prev:i])
            #print(prev, i)
            st = np.where(df['total_AP'][prev:i] > 0.2)[0][0] + prev
            end = np.where(df['total_AP'].loc[prev:i] > 0.2)[0][len(np.where(df['total_AP'].loc[prev:i] > 0.2)[0]) - 1] + prev
            emd = EMD()
            emd.emd(np.array(df['total_AP'][st:end]))
            imfs, res = emd.get_imfs_and_residue()
            #print(st, end, k)
            #if k == 11:
                #return df['total_AP'][st:end]
            try:
                df['total_AP'][st:end] = imfs[2] + df['total_AP'][st:end].mean()
            except IndexError:
                try:
                    df['total_AP'][st:end] = imfs[1] + df['total_AP'][st:end].mean()
                except IndexError:
                    df['total_AP'][st:end] = imfs[0] + df['total_AP'][st:end].mean()
            #df_help = pd.concat((df_help, df.loc[st:end]), axis=0)
            prev = i
            
            k += 1

    if mode == 'original':
        if dimension == 'one':
            return np.array(df['total_AP'])
        elif dimension == 'multi':
            df['hour'] = df['event_timestamp'].dt.hour
            #df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            #df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'weekday']])
            return X
        elif dimension == 'multi_intervals':
            X = []
            
            for raw in range(len(df)):
                X_raw = []
                X_raw.append(df.total_AP[raw])
                weekday = df['event_timestamp'][raw].dayofweek
                for i in range(7):
                    if i == weekday:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                hour = df['event_timestamp'][raw].hour
                for i in range(24):
                    if i == hour:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                minute = df['event_timestamp'][raw].minute
                for i in range(60 // intervals):
                    if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                X.append(X_raw)
            return np.array(X)
            
            
    elif mode == 'only_days':
        df = df[(df.event_timestamp.apply(datetime.time) < datetime.strptime('19:10:00', '%H:%M:%S').time()) & \
             (df.event_timestamp.apply(datetime.time) > datetime.strptime('6:40:00', '%H:%M:%S').time())].reset_index(drop=True)
        if dimension == 'one':
            return np.array(df['total_AP'])
        elif dimension == 'multi':
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'day', 'weekday', 'month']])
            return X
        elif dimension == 'multi_intervals':
            X = []
            
            for raw in range(len(df)):
                X_raw = []
                X_raw.append(df.total_AP[raw])
                weekday = df['event_timestamp'][raw].dayofweek
                for i in range(7):
                    if i == weekday:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                hour = df['event_timestamp'][raw].hour
                for i in range(24):
                    if i == hour:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                minute = df['event_timestamp'][raw].minute
                for i in range(60 // intervals):
                    if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                X.append(X_raw)
            return np.array(X)
    elif mode == 'windows':
        ind = np.where(df.event_timestamp.apply(datetime.time) > datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        ans = np.zeros((len(ind), 720))
        prev = ind[0]
        k = 0
        if dimension == 'one':
            #print(ind)
            for i in ind[1:]:
                #print(i)
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                #print(day)
                #print(day.shape)
                #print(prev, i, k)
                ans[k] = day
                prev = i
                k += 1
            return np.array(ans)[:-1]#.reshape(7, 720)
        elif dimension == 'multi':
            ans = np.zeros((len(ind), 720, 5))
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            k = 0
            for i in ind[1:]:
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                        hour = np.array(df['hour'].loc[prev:i])\
                        [shape_ - 720:]
                        day_ = np.array(df['day'].loc[prev:i])\
                        [shape_ - 720:]
                        weekday = np.array(df['weekday'].loc[prev:i])\
                        [shape_ - 720:]
                        month = np.array(df['month'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        hour = np.array(df['hour'].loc[prev - (720 - shape_):i])
                        day_ = np.array(df['day'].loc[prev - (720 - shape_):i])
                        weekday = np.array(df['weekday'].loc[prev - (720 - shape_):i])
                        month = np.array(df['month'].loc[prev - (720 - shape_):i])
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                    hour = np.array(df['hour'].loc[prev:i])
                    day_ = np.array(df['day'].loc[prev:i])
                    weekday = np.array(df['weekday'].loc[prev:i])
                    month = np.array(df['month'].loc[prev:i])
                
                ans[k] = np.concatenate((day[:, np.newaxis], hour[:, np.newaxis], day_[:, np.newaxis]\
                                         , weekday[:, np.newaxis], month[:, np.newaxis]), axis=1)
                prev = i
                k += 1
            return np.array(ans)[:-1]
        elif dimension == 'multi_intervals':
            X = []
            print(ind)
            for j in ind[1:]:
                #print(prev, j, j - prev)
                X_day = []
                df_day = df.loc[prev:j - 1].reset_index(drop=True)
                #print(len(df_day))
                
                if len(df_day) > 720:
                    print('BIG')
                    df_day = df_day.loc[:720].reset_index(drop=True)
                    #print(len(df_day))
                elif len(df_day) < 720:
                    print('SMALL')
                    print(len(df))
                    if len(df) < 720*2:
                        print('HIII')
                        df_day = df.loc[prev - (720 - j + prev) :j - 1].reset_index(drop=True)
                    else:
                        df_day = df.loc[prev:(720+ prev)-1].reset_index(drop=True)
                print(len(df_day))
                for raw in range(len(df_day)):
                    X_raw = []
                    X_raw.append(df_day.total_AP[raw])
                    weekday = df_day['event_timestamp'][raw].dayofweek
                    for i in range(7):
                        if i == weekday:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    hour = df_day['event_timestamp'][raw].hour
                    for i in range(24):
                        if i == hour:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    minute = df_day['event_timestamp'][raw].minute
                    for i in range(60 // intervals):
                        if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    X_day += X_raw
                X.append(X_day)
                #print(prev, j)
                prev = j
            return np.array(X)
    else:
        print('INCORRECT MODE')
        return None
    
def get_data1(df, mode='original', emd=False, dimension='one', window_size=0):
    delta_seconds = df.event_timestamp.diff() / np.timedelta64(1, 's')
    delta_seconds[np.where(delta_seconds == 0)[0]] = 1e-3
    delta_seconds = delta_seconds[1:]

    AP = df['Total_AP_energy_max']

    a = []
    for j, i in enumerate(df['Total_AP_energy_max']):
        try:
            i = float(i)
        except ValueError:
            a.append(j)
    for i in a:
        AP[i] = AP[i - 1]

    AP = np.diff(np.array(AP, dtype = float)) * 1000
    X = AP / delta_seconds
    df['total_AP'] = X
    if emd:
        ind = np.where(df.event_timestamp.apply(datetime.time) >= datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        
        #print(ind)
        prev = 0
        #df_help = pd.DataFrame(columns=df.columns)
        k = 0
        for i in ind:
            #print(i)
            #print(df.loc[prev:i])
            #print(prev, i)
            st = np.where(df['total_AP'][prev:i] > 0.2)[0][0] + prev
            end = np.where(df['total_AP'].loc[prev:i] > 0.2)[0][len(np.where(df['total_AP'].loc[prev:i] > 0.2)[0]) - 1] + prev
            emd = EMD()
            emd.emd(np.array(df['total_AP'][st:end]))
            imfs, res = emd.get_imfs_and_residue()
            #print(st, end, k)
            #if k == 11:
                #return df['total_AP'][st:end]
            try:
                df['total_AP'][st:end] = imfs[2] + df['total_AP'][st:end].mean()
            except IndexError:
                try:
                    df['total_AP'][st:end] = imfs[1] + df['total_AP'][st:end].mean()
                except IndexError:
                    df['total_AP'][st:end] = imfs[0] + df['total_AP'][st:end].mean()
            #df_help = pd.concat((df_help, df.loc[st:end]), axis=0)
            prev = i
            
            k += 1

    if mode == 'original':
        if dimension == 'one':
            return np.array(df['total_AP'])
        elif dimension == 'multi':
            df['hour'] = df['event_timestamp'].dt.hour
            #df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            #df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'weekday']])
            return X
        elif dimension == 'multi_intervals':
            X = []
            
            for raw in range(len(df)):
                X_raw = []
                X_raw.append(df.total_AP[raw])
                weekday = df['event_timestamp'][raw].dayofweek
                for i in range(7):
                    if i == weekday:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                hour = df['event_timestamp'][raw].hour
                for i in range(24):
                    if i == hour:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                minute = df['event_timestamp'][raw].minute
                for i in range(60 // intervals):
                    if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                X.append(X_raw)
            return np.array(X)
    elif mode == 'only_days':
        df = df[(df.event_timestamp.apply(datetime.time) < datetime.strptime('19:10:00', '%H:%M:%S').time()) & \
             (df.event_timestamp.apply(datetime.time) > datetime.strptime('6:40:00', '%H:%M:%S').time())].reset_index(drop=True)
        if dimension == 'one':
            return np.array(df['total_AP'])
        else:
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'day', 'weekday', 'month']])
            return X
    elif mode == 'windows':
        ind = np.where(df.event_timestamp.apply(datetime.time) > datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        ans = np.zeros((len(ind), 720))
        prev = ind[0]
        k = 0
        if dimension == 'one':
            print(ind)
            for i in ind[1:]:
                #print(i)
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                #print(day)
                #print(day.shape)
                #print(prev, i, k)
                ans[k] = day
                prev = i
                k += 1
            return np.array(ans)[:-1]#.reshape(7, 720)
        elif dimension == 'multi':
            ans = np.zeros((len(ind), 720, 5))
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            k = 0
            for i in ind[1:]:
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                        hour = np.array(df['hour'].loc[prev:i])\
                        [shape_ - 720:]
                        day_ = np.array(df['day'].loc[prev:i])\
                        [shape_ - 720:]
                        weekday = np.array(df['weekday'].loc[prev:i])\
                        [shape_ - 720:]
                        month = np.array(df['month'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        hour = np.array(df['hour'].loc[prev - (720 - shape_):i])
                        day_ = np.array(df['day'].loc[prev - (720 - shape_):i])
                        weekday = np.array(df['weekday'].loc[prev - (720 - shape_):i])
                        month = np.array(df['month'].loc[prev - (720 - shape_):i])
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                    hour = np.array(df['hour'].loc[prev:i])
                    day_ = np.array(df['day'].loc[prev:i])
                    weekday = np.array(df['weekday'].loc[prev:i])
                    month = np.array(df['month'].loc[prev:i])
                
                ans[k] = np.concatenate((day[:, np.newaxis], hour[:, np.newaxis], day_[:, np.newaxis]\
                                         , weekday[:, np.newaxis], month[:, np.newaxis]), axis=1)
                prev = i
                k += 1
            return np.array(ans)[:-1]
        elif dimension == 'multi_intervals':
            X = []
            print(ind)
            for j in ind[1:]:
                #print(prev, j, j - prev)
                X_day = []
                df_day = df.loc[prev:j - 1].reset_index(drop=True)
                #print(len(df_day))
                #print(df_day)
                if len(df_day) > 720:
                    #print('BIG')
                    df_day = df_day.loc[:720].reset_index(drop=True)
                    #print(len(df_day))
                elif len(df_day) < 720:
                    #print('SMALL')
                    df_day = df.loc[prev:j + (720 - j + prev)].reset_index(drop=True)
                for raw in range(len(df_day)):
                    X_raw = []
                    X_raw.append(df_day.total_AP[raw])
                    weekday = df_day['event_timestamp'][raw].dayofweek
                    for i in range(7):
                        if i == weekday:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    hour = df_day['event_timestamp'][raw].hour
                    for i in range(24):
                        if i == hour:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    minute = df_day['event_timestamp'][raw].minute
                    for i in range(60 // intervals):
                        if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    X_day += X_raw
                X.append(X_day)
                #print(prev, j)
                prev = j
            return np.array(X)
                
    else:
        print('INCORRECT MODE')
        return None



def get_data_3(df, mode='original', emd=False, dimension='one', window_size=0):
    delta_seconds = df.event_timestamp.diff() / np.timedelta64(1, 's')
    delta_seconds[np.where(delta_seconds == 0)[0]] = 1e-3
    delta_seconds = delta_seconds[1:]

    AP = df['Total_AP_energy_max']

    a = []
    for j, i in enumerate(df['Total_AP_energy_max']):
        try:
            i = float(i)
        except ValueError:
            a.append(j)
    for i in a:
        AP[i] = AP[i - 1]

    AP = np.diff(np.array(AP, dtype = float)) * 1000
    X = AP / delta_seconds
    df['total_AP'] = X
    if emd:
        ind = np.where(df.event_timestamp.apply(datetime.time) >= datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        
        #print(ind)
        prev = 0
        #df_help = pd.DataFrame(columns=df.columns)
        k = 0
        for i in ind[1:]:
            print(i)
            #print(df.loc[prev:i])
            print(prev, i)
            st = np.where(df['total_AP'][prev:i] > 0.5)[0][0] + prev
            end = np.where(df['total_AP'].loc[prev:i] > 0.5)[0][len(np.where(df['total_AP'].loc[prev:i] > 0.5)[0]) - 1] + prev
            emd = EMD()
            emd.emd(np.array(df['total_AP'][st:end]))
            imfs, res = emd.get_imfs_and_residue()
            print(st, end, k)
            #if k == 11:
                #return df['total_AP'][st:end]
            try:
                df['total_AP'][st:end] = imfs[2] + df['total_AP'][st:end].mean()
            except IndexError:
                try:
                    df['total_AP'][st:end] = imfs[1] + df['total_AP'][st:end].mean()
                except IndexError:
                    df['total_AP'][st:end] = imfs[0] + df['total_AP'][st:end].mean()
            #df_help = pd.concat((df_help, df.loc[st:end]), axis=0)
            prev = i
            
            k += 1

    if mode == 'original':
        if dimension == 'one':
            return np.array(df['total_AP'])
        elif dimension == 'multi':
            df['hour'] = df['event_timestamp'].dt.hour
            #df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            #df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'weekday']])
            return X
        elif dimension == 'multi_intervals':
            X = []
            
            for raw in range(len(df)):
                X_raw = []
                X_raw.append(df.total_AP[raw])
                weekday = df['event_timestamp'][raw].dayofweek
                for i in range(7):
                    if i == weekday:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                hour = df['event_timestamp'][raw].hour
                for i in range(24):
                    if i == hour:
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                minute = df['event_timestamp'][raw].minute
                for i in range(60 // intervals):
                    if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                        X_raw.append(1)
                    else:
                        X_raw.append(0)
                X.append(X_raw)
            return np.array(X)
    elif mode == 'only_days':
        df = df[(df.event_timestamp.apply(datetime.time) < datetime.strptime('19:10:00', '%H:%M:%S').time()) & \
             (df.event_timestamp.apply(datetime.time) > datetime.strptime('6:40:00', '%H:%M:%S').time())].reset_index(drop=True)
        if dimension == 'one':
            return np.array(df['total_AP'])
        else:
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            X = np.array(df[['total_AP', 'hour', 'day', 'weekday', 'month']])
            return X
    elif mode == 'windows':
        ind = np.where(df.event_timestamp.apply(datetime.time) > datetime.strptime('23:58:00', '%H:%M:%S').time())[0]
        ans = np.zeros((len(ind), 720))
        prev = ind[0]
        k = 0
        if dimension == 'one':
            print(ind)
            for i in ind[1:]:
                #print(i)
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                #print(day)
                #print(day.shape)
                print(prev, i, k)
                ans[k] = day
                prev = i
                k += 1
            return np.array(ans)[:-1]#.reshape(7, 720)
        elif dimension == 'multi':
            ans = np.zeros((len(ind), 720, 5))
            df['hour'] = df['event_timestamp'].dt.hour
            df['day'] = df['event_timestamp'].dt.day
            df['weekday'] = df['event_timestamp'].dt.dayofweek
            df['month'] = df['event_timestamp'].dt.month
            k = 0
            for i in ind[1:]:
                shape_ = np.array(df['total_AP'].loc[prev:i])[:, np.newaxis].shape[0]
                if shape_ != 720:
                    if shape_ > 720:
                        day = np.array(df['total_AP'].loc[prev:i])\
                        [shape_ - 720:]
                        hour = np.array(df['hour'].loc[prev:i])\
                        [shape_ - 720:]
                        day_ = np.array(df['day'].loc[prev:i])\
                        [shape_ - 720:]
                        weekday = np.array(df['weekday'].loc[prev:i])\
                        [shape_ - 720:]
                        month = np.array(df['month'].loc[prev:i])\
                        [shape_ - 720:]
                    else:
                        day = np.array(df['total_AP'].loc[prev - (720 - shape_):i])
                        hour = np.array(df['hour'].loc[prev - (720 - shape_):i])
                        day_ = np.array(df['day'].loc[prev - (720 - shape_):i])
                        weekday = np.array(df['weekday'].loc[prev - (720 - shape_):i])
                        month = np.array(df['month'].loc[prev - (720 - shape_):i])
                else:
                    day = np.array(df['total_AP'].loc[prev:i])
                    hour = np.array(df['hour'].loc[prev:i])
                    day_ = np.array(df['day'].loc[prev:i])
                    weekday = np.array(df['weekday'].loc[prev:i])
                    month = np.array(df['month'].loc[prev:i])
                
                ans[k] = np.concatenate((day[:, np.newaxis], hour[:, np.newaxis], day_[:, np.newaxis]\
                                         , weekday[:, np.newaxis], month[:, np.newaxis]), axis=1)
                prev = i
                k += 1
            return np.array(ans)[:-1]
        elif dimension == 'multi_intervals':
            X = []
            print(ind)
            for j in ind[1:]:
                #print(prev, j, j - prev)
                X_day = []
                df_day = df.loc[prev:j - 1].reset_index(drop=True)
                #print(len(df_day))
                #print(df_day)
                if len(df_day) > 720:
                    #print('BIG')
                    df_day = df_day.loc[:720].reset_index(drop=True)
                    #print(len(df_day))
                elif len(df_day) < 720:
                    #print('SMALL')
                    df_day = df.loc[prev:j + (720 - j + prev)].reset_index(drop=True)
                for raw in range(len(df_day)):
                    X_raw = []
                    X_raw.append(df_day.total_AP[raw])
                    weekday = df_day['event_timestamp'][raw].dayofweek
                    for i in range(7):
                        if i == weekday:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    hour = df_day['event_timestamp'][raw].hour
                    for i in range(24):
                        if i == hour:
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    minute = df_day['event_timestamp'][raw].minute
                    for i in range(60 // intervals):
                        if (minute >= (i * intervals)) and (minute < ((i + 1) * intervals)):
                            X_raw.append(1)
                        else:
                            X_raw.append(0)
                    X_day += X_raw
                X.append(X_day)
                #print(prev, j)
                prev = j
            return np.array(X)
                
    else:
        print('INCORRECT MODE')
        return Nonea

