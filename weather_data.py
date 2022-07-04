#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:34:18 2020

@author: yiye
"""
import numpy as np
import pandas as pd
import os
import csv
import pickle

# Pre-detrend control
isdetrend = False

station_coordinate = pd.DataFrame(np.zeros(shape = (1218, 3)), columns=['ID', 'lat', 'long'])
with open('Weather data/stations.txt', 'r') as fd:
    reader = csv.reader(fd)
    for s, row in enumerate(reader):
        station_coordinate.iloc[s,0] = row[0][:11]
        station_coordinate.iloc[s,1] = float(row[0][12:20])
        station_coordinate.iloc[s,2] = float(row[0][21:30])
station_coordinate.sort_values(by='ID', inplace = True)

eff_yr = []
with open('Weather data/tmin/' + 'USH00011084.FLs.52j.tmin', 'r') as fd:
    reader = csv.reader(fd)
    for y, row in enumerate(reader):
        if y in range(3): continue
        eff_yr.append(row[0][12:16])
T = 127*12-1

station_list_tmin = os.listdir('Weather data/tmin/')
station_list_tmin.remove('.DS_Store')
station_list_tmin.sort()
tmin = np.zeros(shape = (1218, T))      
for i, station in enumerate(station_list_tmin):
    #if station[5:7] != '04': continue # take the stations of california
    with open('Weather data/tmin/' + station, 'r') as fd:
        reader = csv.reader(fd)
        table =  np.zeros(shape = (127, 12))
        y = 0
        for row in reader:
            if row[0][12:16] not in eff_yr: continue
            if row[0][12:16] != eff_yr[y]: 
                table[:,:] = np.nan
                break
            for m in range(12):
                table[y, m] = int(row[0][(17+9*m-1):(17+9*m+5)])/100
            y += 1
        a = table.copy()
        table[table == -99.99] = np.nan
        if isdetrend: table -= np.nanmean(table,axis = 0) # detrend using monthly trends 
        tmin[i,:] = table.flatten()[:-1]
std_tmin = np.nanstd(tmin, axis = 1)


station_list_tmax = os.listdir('Weather data/tmax/')
station_list_tmax.sort()
tmax = np.zeros(shape = (1218, T))        
for i, station in enumerate(station_list_tmax):
    #if station[5:7] != '04': continue # take the stations of california
    with open('Weather data/tmax/' + station, 'r') as fd:
        reader = csv.reader(fd)
        table =  np.zeros(shape = (127, 12))
        y = 0
        for row in reader:
            if row[0][12:16] not in eff_yr: continue
            if row[0][12:16] != eff_yr[y]: 
                table[:,:] = np.nan
                break
            for m in range(12):
                table[y, m] = int(row[0][(17+9*m-1):(17+9*m+5)])/100
            y += 1
        table[table == -99.99] = np.nan
        if isdetrend: table -= np.nanmean(table,axis = 0)   
        tmax[i,:] = table.flatten()[:-1]
std_tmax = np.nanstd(tmax, axis = 1)

        
station_list_tavg = os.listdir('Weather data/tavg/')
station_list_tavg.sort()
tavg = np.zeros(shape = (1218, T))        
for i, station in enumerate(station_list_tavg):
    #if station[5:7] != '04': continue # take the stations of california
    with open('Weather data/tavg/' + station, 'r') as fd:
        reader = csv.reader(fd)
        table =  np.zeros(shape = (127, 12))
        y = 0
        for row in reader:
            if row[0][12:16] not in eff_yr: continue
            if row[0][12:16] != eff_yr[y]: 
                table[:,:] = np.nan
                break
            for m in range(12):
                table[y, m] = int(row[0][(17+9*m-1):(17+9*m+5)])/100
            y += 1
        table[table == -99.99] = np.nan
        if isdetrend: table -= np.nanmean(table,axis = 0)        
        tavg[i,:] = table.flatten()[:-1]
std_tavg = np.nanstd(tavg, axis = 1)


station_list_prcp = os.listdir('Weather data/prcp/')
station_list_prcp.sort()
prcp = np.zeros(shape = (1218, T))        
for i, station in enumerate(station_list_prcp):
    #if station[5:7] != '04': continue # take the stations of california
    with open('Weather data/prcp/' + station, 'r') as fd:
        reader = csv.reader(fd)
        table =  np.zeros(shape = (127, 12))
        y = 0
        for row in reader:
            if row[0][12:16] not in eff_yr: continue
            if row[0][12:16] != eff_yr[y]: 
                table[:,:] = np.nan
                break
            for m in range(12):
                table[y, m] = int(row[0][(17+9*m-1):(17+9*m+5)])/100
            y += 1
        table[table == -99.99] = np.nan
        if isdetrend: table -= np.nanmean(table,axis = 0)        
        prcp[i,:] = table.flatten()[:-1]
std_prcp = np.nanstd(prcp, axis = 1)


a = ~np.isnan(tmin).any(axis=1)
b = ~np.isnan(tmax).any(axis=1)
c = ~np.isnan(tavg).any(axis=1)
d = ~np.isnan(prcp).any(axis=1)

eff_station = a*b*c*d

# Take California No.04 and Nevada No.26
ind04 = [i for i, x in enumerate(station_list_tmin) if eff_station[i] and x[5:7] == '04']
ind26 = [i for i, x in enumerate(station_list_tmin) if eff_station[i] and x[5:7] == '26']
station_name = [x for i, x in enumerate(station_list_tmin) if i in (ind04 + ind26)]
 
ind = ind04 + ind26

dt_MTS = np.zeros(shape = (len(ind), 4, T))   
for t in range(T):
    dt_MTS[:, 0, t] = tmin[ind, t]
    dt_MTS[:, 1, t] = tmax[ind, t]
    dt_MTS[:, 2, t] = tavg[ind, t]
    dt_MTS[:, 3, t] = prcp[ind, t]

with open('dt_MTS_04_26','wb') as fp:
    pickle.dump((dt_MTS, ind),fp)
        
    
#dt_MTS = np.zeros(shape = (len(ind), 4, T))   
#for t in range(T):
#    dt_MTS[:, 0, t] = tmin[ind, t]/std_tmin[ind]
#    dt_MTS[:, 1, t] = tmax[ind, t]/std_tmax[ind]
#    dt_MTS[:, 2, t] = tavg[ind, t]/std_tavg[ind]
#    dt_MTS[:, 3, t] = prcp[ind, t]/std_prcp[ind]
    