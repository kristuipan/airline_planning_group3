#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np


# In[14]:


#import airport data
df_airports = pd.read_excel('AE4423Ass2/AirportData.xlsx', header=None).T
df_airports.columns = df_airports.iloc[0]
df_airports = df_airports[1:]

#convert to arrays
city = df_airports['City'].to_numpy()
iata = df_airports['IATA code'].to_numpy()
latitude = df_airports['Latitude (deg)'].to_numpy()
longitude = df_airports['Longitude (deg)'].to_numpy()
runway = df_airports['Runway (m)'].to_numpy()


# In[69]:


#import aircraft data
df_aircraft = pd.read_excel('AE4423Ass2/FleetType.xlsx', header=None)
df_aircraft.columns = df_aircraft.iloc[0]
df_aircraft = df_aircraft[1:]

#convert to arrays
speed = df_aircraft.iloc[0,1:].to_numpy()
capacity = df_aircraft.iloc[1,1:].to_numpy()
tat = df_aircraft.iloc[2,1:].to_numpy()
max_range = df_aircraft.iloc[3,1:].to_numpy()
req_runway = df_aircraft.iloc[4,1:].to_numpy()
lease_cost = df_aircraft.iloc[5,1:].to_numpy()
fixed_operating_cost = df_aircraft.iloc[6,1:].to_numpy()
hourly_cost = df_aircraft.iloc[7,1:].to_numpy()
fuel_cost_parameter = df_aircraft.iloc[8,1:].to_numpy()
fleet = df_aircraft.iloc[9,1:].to_numpy()


# In[38]:


#import demand data
df_demand = pd.read_excel('AE4423Ass2/Group3.xlsx', header=None)

#look up demand for airport pair and time window
def demand(ap1, ap2, tw):
    return float(df_demand[(df_demand[1] == ap1) & (df_demand[2] == ap2)][tw+3])


# In[41]:


latitude


# In[52]:


#calculate distance between airport pairs
iata_list = list(iata)
def distance(ap1, ap2):
    R_E = 6371 #km
    phi1 = np.radians(latitude[iata_list.index(ap1)])
    phi2 = np.radians(latitude[iata_list.index(ap2)])
    lamda1 = np.radians(longitude[iata_list.index(ap1)])
    lamda2 = np.radians(longitude[iata_list.index(ap2)])
    sigma = 2* np.arcsin(np.sqrt(np.sin((phi1 - phi2)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin((lamda1 - lamda2)/2)**2))
    d_ij = R_E * sigma
    return float(d_ij)    


# In[63]:


#calculate revenue for flight leg (i,j) with aircraft type k at time t
def revenue(ap1, ap2, k, t):
    yield_value = 0.26
    time_window = int(t/40)
    #check if demand fits in the aircraft
    if demand(ap1, ap2, time_window) <= capacity[k]:
        flow = demand(ap1, ap2, time_window)
    else:
        flow = capacity[k]
    #convert flow to tonnes
    flow = flow/1000
    revenue = yield_value * distance(ap1, ap2) * flow
    return revenue


# In[74]:


#calculate cost for flight leg (i,j) with aircraft type k
def cost(ap1, ap2, k):
    time_based_cost = hourly_cost[k] * (distance(ap1, ap2)/speed[k] + 0.5) #include take-off and landing time
    f = 1.42
    fuel_cost = fuel_cost_parameter[k] * f * distance(ap1, ap2) / 1.5
    total_cost = time_based_cost + fuel_cost + fixed_operating_cost[k]
    return total_cost


# In[83]:


#initialize sets
I = range(20) #airports
T = range(1200) #time steps
K = range(3) #aircraft type


# In[97]:


#calculate possible arc weights (time) to and from hub in time steps
flight_time = np.zeros((20,3))
for i in I:
    for k in K:
        flight_time[i][k] = np.ceil((distance(iata[4], iata[i])/speed[k] + 0.5 + 0.5*tat[k]/60)*10) #include tat in arc weight


# In[117]:


dp = {airport: {t: {aircraft: -float('inf') for aircraft in K} for t in range(1200)} for airport in iata}
for aircraft in K:
    dp['MAD'][0][aircraft] = 0

for t in range(1,1200):
    for current_airport in iata:
        for aircraft in K:
            if dp[current_airport][t][aircraft] != -float('inf'):
                for next_airport in iata:
                    if next_airport != current_airport:
                        flight_duration = get_flight_duration(current_airport, next_airport, aircraft)
                        if t + flight_duration <= 1200:
                            profit = calculate_profit(current_airport, next_airport, aircraft)
                            dp[next_airport][t + flight_duration][aircraft] = max(
                                dp[next_airport][t + flight_duration][aircraft],
                                dp[current_airport][t][aircraft] + profit
                            )


# In[120]:


# Initialize DP table
dp = {('MAD', 0, aircraft, True): 0 for aircraft in K}

# Iterate over time steps
for t in range(1200):
    for current_airport in iata:
        for aircraft in K:
            for at_hub in [True, False]:
                if (current_airport, t, aircraft, at_hub) in dp:
                    current_profit = dp[(current_airport, t, aircraft, at_hub)]
                    if at_hub:
                        # Depart from hub to another airport
                        for next_airport in iata:
                            if next_airport != 'MAD':
                                flight_duration = flight_time[iata_list.index(next_airport)][aircraft]
                                if t + flight_duration <= 1200:
                                    profit = revenue('MAD', next_airport, t, aircraft) - cost('MAD', next_airport, aircraft)
                                    next_state = (next_airport, t + flight_duration, aircraft, False)
                                    dp[next_state] = max(dp.get(next_state, float('-inf')),
                                                         current_profit + profit)
                    else:
                        # Return to hub from current airport
                        flight_duration = flight_time[iata_list.index(next_airport)][aircraft]
                        if t + flight_duration <= 1200:
                            profit = calculate_profit(current_airport, 'MAD', aircraft)
                            next_state = ('MAD', t + flight_duration, aircraft, True)
                            dp[next_state] = max(dp.get(next_state, float('-inf')),
                                                 current_profit + profit)

# Extract maximum profit at the hub at the final time step
max_profit = max(dp.get(('MAD', 1200, aircraft, True), float('-inf')) for aircraft in K)
print(f"Maximum Profit: {max_profit}")


# In[106]:


K


# In[ ]:




