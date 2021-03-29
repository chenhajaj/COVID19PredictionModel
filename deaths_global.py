

import pandas as pd
import numpy as np

def statePos(df):
    """""
    Mapping between states and their coordinates into a dictionary
    Parameters: A data frame which contain the data
    Return: Coordinate's dictionary
    note: France, Denmark added manually as the have more than one occurrence
    """""
    coordinates = dict()
    for state in df:
        if state == 'France':
            coordinates[state] = (46.227638, 2.213749)
        elif state == 'Denmark':
            coordinates[state] = (56.2639, 9.5018)
        elif state == "Netherlands":
            coordinates[state] = (52.1326,5.2913)
        
    return coordinates

def confirmedFromGit():
    """""
    url: Data from GitHub
    Return: Daily updated Data frame which contain COVID-19 confirmed 
    """""
    url = url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
          'csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    df = pd.read_csv(url, error_bad_lines=False, header=0)
    df = df.groupby(['Country/Region']).sum().T
    gdict = statePos(df)

    df = df.drop(['Lat', 'Long'])
    df = df[states]
    df.to_csv('DeathTimeSeries.csv')
    
    
    return df, gdict


def expSmoothing(a, i, c_r):
    """""
    Generate exponential smoothing for confirmed seeks in the state 
    Parameters: 
    a - alpha 
    i - The initial period
    c_r - Vector of observations
    Return: Vector of smoothing values
    """""

    smoothing_values = []
    for t in range(len(c_r)):
        init = []
        if t == 0 or t == 1:
            smoothing_values.append(c_r[t])
            init.append(pow((1 - a), i - t) * c_r[t])
        elif t == i:
            init.append(pow(a * (1 - a), i - t) * c_r[t])
            smoothing_values.append(a * sum(init) + (1 - a) * c_r[0] * t)
        else:
            smoothing_values.append(a * c_r[t] + (1 - a) * smoothing_values[t - 1])

    return smoothing_values

def weeklyMean(outdf, indf):
    low_pos, up_pos = 0, 7
    while up_pos < len(indf.index):
        week_mean = pd.DataFrame(indf.iloc[low_pos:up_pos, :].mean(axis=0)).T
        week_mean['date'] = indf['date'].values[low_pos]
        outdf = pd.concat([outdf, week_mean], ignore_index=True, sort=False)
        low_pos, up_pos = low_pos + 7, up_pos + 7
    return outdf
    
    
    
    
states=[  'Germany', 'Italy', 'Spain', 'Belgium', 'Switzerland', 'Austria',
           'France',   'Finland', 'Greece', 'Netherlands']

"""death daily"""

confirmed_ts, geo_dict = confirmedFromGit()

"""generate exponential smoothing:"""

d_t = {}
f_t = {}
index = {}
I = 2
alpha = 0.3
for state in states:
    index[state] = confirmed_ts[state].index
    d_t[state] = np.asarray(confirmed_ts[state])
    f_t[state] = expSmoothing(alpha, I, d_t[state])

daily_f_t = pd.DataFrame.from_dict(f_t, orient='columns')
daily_f_t.index = confirmed_ts.index
daily_f_t.to_csv('DailyForecastExpSmoothingDeaths.csv')


"""generate weekly Exp of Deaths for ever state:"""
file="DailyForecastExpSmoothingDeaths.csv"
df=pd.read_csv(file)
df['date'] = df.index
exp_confirmed_week = pd.DataFrame(columns=df.columns)
exp_confirmed_week = weeklyMean(exp_confirmed_week, df.iloc[4:, :])
exp_confirmed_week.to_csv('ExpWeeklyDeaths.csv')


"""generate weekly Mean of Deaths for ever state:"""
confirmed_ts['date'] = confirmed_ts.index
mean_confirmed_week = pd.DataFrame(columns=confirmed_ts.columns)
mean_confirmed_week = weeklyMean(mean_confirmed_week, confirmed_ts.iloc[4:, :])
mean_confirmed_week.to_csv('MeanWeeklyDeaths.csv')



