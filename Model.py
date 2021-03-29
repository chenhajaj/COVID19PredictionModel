import pandas as pd
import numpy as np
import folium
from collections import namedtuple



def statePos(df):
    """""
    Mapping between states and their coordinates into a dictionary
    Parameters: A data frame which contain the data
    Return: Coordinate's dictionary
    note: France, Denmark,Netherlands  added manually as the have more than one occurrence
    """""
    coordinates = dict()
    for state in df:
        if state == 'France':
            coordinates[state] = (46.227638, 2.213749)
        elif state == 'Denmark':
            coordinates[state] = (56.2639, 9.5018)
        elif state == "Netherlands":
            coordinates[state] = (52.1326,5.2913)
            
        else:
            coordinates[state] = (df[state][0], df[state][1])
    return coordinates


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


def weeklySum(outdf, indf):
    low_pos, up_pos = 0, 7
    while up_pos < len(indf.index):
        week_mean = pd.DataFrame(indf.iloc[low_pos:up_pos, :].sum(axis=0)).T
        week_mean['date'] = indf['date'].values[low_pos]
        outdf = pd.concat([outdf, week_mean], ignore_index=True, sort=False)
        low_pos, up_pos = low_pos + 7, up_pos + 7
    return outdf


def distFromCenter(centers, gdict, states):
    """""
    Generate mean weekly distance from center of mass for each state
    distance = sqrt((point_x - center_x)^2 + (point_y - center_y)^2)
    Parameters: 
    centers - df of daily centers 
    gdict - Dictionary of state's coordinates {'state': lat, long}
    states - List of pre-defined states that taking in consideration
    """""
    daily_dist = {}
    for index in centers.index:
        center = np.array(centers['lat'][index], centers['long'][index])
        for key in gdict:
            point = np.array(gdict[key])
            distance = np.linalg.norm(center - point)
            if key in states:
                if key not in daily_dist.keys():
                    daily_dist[key] = [distance]
                else:
                    daily_dist[key].append(distance)
    temp = pd.DataFrame.from_dict(daily_dist, orient='columns')
    daily_dist = pd.concat([centers, temp], axis=1, sort=False).iloc[4:, :]
    mean_dist_week = pd.DataFrame(columns=daily_dist.columns)
    mean_dist_week = weeklyMean(mean_dist_week, daily_dist)
    mean_dist_week.to_csv('MeanWeeklyDistance.csv')


def ICoM(gdict, tseries, c_r, states):
    """""
    Generate center of mass for every day
    center of mass = (∑_(∀r) (D(r) )⃗∙C_(r,t))/(∑_(∀r) C_(r,t))
    Parameters: 
    gdict - Dictionary of state's coordinates {'state': lat, long}
    tseries - df of daily confirmed for each state
    c_r - Dictionary of daily confirmed for each country {'state': confirmed start date,...,confirmed end date)
    states - List of pre-defined states that taking in consideration
    """""
    icom = []
    for i in range(len(tseries.index)):
        numerator = []
        denominator = []
        
        for state in states:
            coordinates = np.array([gdict[state][0], gdict[state][1]])
            numerator.append(c_r[state][i] * coordinates)
            

            denominator.append(c_r[state][i])
        numerator = np.sum(np.array(numerator), axis=0)
        denominator = sum(denominator)
        if denominator == 0:
            icom.append(np.array([0, 0]))
        else:
            icom.append(numerator * (1/denominator))
    icom = np.array(icom)
    icom = pd.DataFrame({'date': tseries.index, 'lat': icom[:, 0], 'long': icom[:, 1]})
    distFromCenter(icom, gdict, states)
    
    

    """
    # weekly ICOM
    icom.date=pd.to_datetime(icom.date)
    icom=icom.resample('W', on='date').mean()
    icom=icom.reset_index()
    icom.to_csv("icom.csv")
    """
    mapICoM(icom, gdict, states)


def get_arrows(locations, color='blue', size=6, n_arrows=3):

    """""
    Get a list of correctly placed and rotated
    arrows/markers to be plotted

    Parameters
    locations : list of lists of lat lons that represent the
                start and end of the line.
                eg [[41.1132, -96.1993],[41.3810, -95.8021]]
    arrow_color : default is 'blue'
    size : default is 6
    n_arrows : number of arrows to create.  default is 3
    Return
    list of arrows/markers
    """""

    Point = namedtuple('Point', field_names=['lat', 'lon'])

    # creating point from our Point named tuple
    p1 = Point(locations[0][0], locations[0][1])
    p2 = Point(locations[1][0], locations[1][1])

    # getting the rotation needed for our marker.
    # Subtracting 90 to account for the marker's orientation
    # of due East(get_bearing returns North)
    rotation = get_bearing(p1, p2) - 90

    # get an evenly space list of lats and lons for our arrows
    # note that I'm discarding the first and last for aesthetics
    # as I'm using markers to denote the start and end
    arrow_lats = np.linspace(p1.lat, p2.lat, n_arrows + 2)[1:n_arrows+1]
    arrow_lons = np.linspace(p1.lon, p2.lon, n_arrows + 2)[1:n_arrows+1]

    arrows = []

    #creating each "arrow" and appending them to our arrows list
    for points in zip(arrow_lats, arrow_lons):
        arrows.append(folium.RegularPolygonMarker(location=points,
                      fill_color=color, number_of_sides=3,
                      radius=size, rotation=rotation))
    return arrows


def get_bearing(p1, p2):

    '''
    Returns compass bearing from p1 to p2

    Parameters
    p1 : namedtuple with lat lon
    p2 : namedtuple with lat lon

    Return
    compass bearing of type float

    Notes
    Based on https://gist.github.com/jeromer/2005586
    '''

    long_diff = np.radians(p2.lon - p1.lon)

    lat1 = np.radians(p1.lat)
    lat2 = np.radians(p2.lat)

    x = np.sin(long_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2)
        - (np.sin(lat1) * np.cos(lat2)
        * np.cos(long_diff)))
    bearing = np.degrees(np.arctan2(x, y))

    # adjusting for compass bearing
    if bearing < 0:
        return bearing + 360
    return bearing


def mapICoM(icom, gdict, states):
    """""
    Create a world map which include points of the pre-defined states and daily points of center of mass
    Parameters: 
    icom - Data frame including mapped center of mass {date: dates, lat: lats, long:longs}
    gdict - Dictionary of state's coordinates {'state': lat, long}
    states - List of pre-defined states that taking in consideration
    """""
    m = folium.Map(location=[icom['long'].mean(), icom['lat'].mean()],tiles='Stamen Terrain')
    arrow = False
    for state in states:
        folium.Marker([gdict[state][0], gdict[state][1]], icon=folium.Icon(color='red'), popup=state).add_to(m)
    
    for i in range(len(icom)):
        if icom.iloc[i]['lat'] == 0:
            continue
        folium.Marker([icom.iloc[i]['lat'], icom.iloc[i]['long']], popup=icom.iloc[i]['date']).add_to(m)
        if arrow:
            p1 = [icom.iloc[i-1]['lat'], icom.iloc[i-1]['long']]
            p2 = [icom.iloc[i]['lat'], icom.iloc[i]['long']]
            folium.PolyLine(locations=[p1, p2], color='blue').add_to(m)
            arrows = get_arrows(locations=[p1, p2], n_arrows=1)
            for arrow in arrows:
                arrow.add_to(m)
        arrow = True
       
        
       
        
    m.save('IcoMmap.html')


def dailyExpGrwoth(df):
    states = df.columns.tolist()
    exp_growth = {}

    for col_pos in range(len(states)):
        exp_growth[states[col_pos]] = []
        for row_pos in range(len(df.iloc[:, col_pos]) - 1):
            exp_growth[states[col_pos]].append\
                (df.iloc[row_pos + 1, col_pos] / df.iloc[row_pos, col_pos])
    daily_exp_growth = pd.DataFrame.from_dict(exp_growth, orient='columns').replace(np.inf, 0)
    daily_exp_growth.index = df.index[1:]
    daily_exp_growth.to_csv('DailyExpGrowth.csv')

    daily_exp_growth['date'] = df.index[1:]
    mean_exp_growth_week = pd.DataFrame(columns=daily_exp_growth.columns)
    mean_exp_growth_week = weeklyMean(mean_exp_growth_week, daily_exp_growth.iloc[3:, :])
    mean_exp_growth_week.to_csv('MeanWeeklyExpGrowthValues.csv')


def confirmedFromGit():
    """""
    url: Data from GitHub
    Return: Daily updated Data frame which contain COVID-19 confirmed 
    """""
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
          'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df = pd.read_csv(url, error_bad_lines=False, header=0)
    df = df.groupby(['Country/Region']).sum().T
    gdict = statePos(df)

    df = df.drop(['Lat', 'Long'])
    df = df[states]
    df.to_csv('ConfirmedTimeSeries.csv')

    return df, gdict



def weeklyexp(outdf, indf):
    low_pos, up_pos = 0, 7
    while up_pos < len(indf.index):
        week_mean = (pd.DataFrame(indf.iloc[low_pos:up_pos, :].mean(axis=0)).T)
        week_mean['Date'] = indf['Date'].values[low_pos]
        outdf = pd.concat([outdf, week_mean], ignore_index=True, sort=False)
        low_pos, up_pos = low_pos + 7, up_pos + 7
    return outdf


# #==========================================================================================# #
    

"""list of states in the model"""
states=[  'Germany', 'Italy', 'Spain', 'Belgium', 'Switzerland', 'Austria',
           'France',   'Finland', 'Greece', 'Netherlands']

confirmed_ts, geo_dict = confirmedFromGit()

"""generate weekly sum of confirmed for ever state:"""

confirmed_ts['date'] = confirmed_ts.index
sum_confirmed_week = pd.DataFrame(columns=confirmed_ts.columns)
sum_confirmed_week = weeklySum(sum_confirmed_week, confirmed_ts.iloc[4:, :])
sum_confirmed_week.to_csv('SumWeeklyConfirmed.csv')

"""generate weekly Mean of confirmed for ever state:"""
mean_confirmed_week = pd.DataFrame(columns=confirmed_ts.columns)
mean_confirmed_week = weeklyMean(mean_confirmed_week, confirmed_ts.iloc[4:, :])
mean_confirmed_week.to_csv('MeanWeeklyConfirmed.csv')

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
daily_f_t.to_csv('DailyForecastExpSmoothing.csv')
dailyExpGrwoth(daily_f_t)

file="DailyForecastExpSmoothing.csv"
df=pd.read_csv(file)
df['Date'] = df.index
exp_confirmed_week = pd.DataFrame(columns=df.columns)
exp_confirmed_week = weeklyexp(exp_confirmed_week, df.iloc[4:, :])
exp_confirmed_week.to_csv('ExpWeeklyConfirmed.csv')


"""generate daily center of mass:"""
ICoM(geo_dict, confirmed_ts, d_t, states)
geo_df=pd.DataFrame.from_dict(geo_dict,orient="index")












