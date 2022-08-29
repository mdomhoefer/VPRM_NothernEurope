import pandas as pd
import math
from pyhdf.SD import SD, SDC
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import csv

def get_stations_year(year):
    cwd = '/data/co2flux/common/mdomhoef/Oslo/Oslo_analysis/'
    year = year
    StationDataPath = cwd + 'data/HH_Data_small/'

    stations_file = cwd + 'Stations.csv'
    years_file = cwd + 'Station_years.csv'
    
    stations = pd.read_csv(stations_file, sep = ',')
    years_st = pd.read_csv(years_file, index_col= 0) 
    
    #get the years and stations with data of year 
    for station in stations['Station'].unique():
        if years_st.loc[year][station] != 1.0:
            stations = stations.drop(stations[stations['Station'] == station].index)
    snames = stations['Station'].unique()
    
    return stations, snames

def get_stations_tile(H,V):
    cwd = '/data/co2flux/common/mdomhoef/Oslo/Oslo_analysis/'
    # StationDataPath = cwd + 'data/HH_Data_small/'
    stations_file = cwd + 'Stations.csv'
    stations = pd.read_csv(stations_file, sep = ',')
    stations = stations[stations['tile_v']==V]
    stations = stations[stations['tile_h']==H]
    snames = stations['Station'].unique()

    return stations, snames 

def get_stations_tile_year(year, H,V): 
    cwd = '/data/co2flux/common/mdomhoef/Oslo/Oslo_analysis/'
    year = year
    StationDataPath = cwd + 'data/HH_Data_small/'

    stations_file = cwd + 'Stations.csv'
    years_file = cwd + 'Station_years.csv'
    
    stations = pd.read_csv(stations_file, sep = ',')
    years_st = pd.read_csv(years_file, index_col= 0) 
    

    #get the years and stations with data of year 
    for station in stations['Station'].unique():
        if years_st.loc[year][station] != 1.0:
            stations = stations.drop(stations[stations['Station'] == station].index)
    
    stations = stations[stations['tile_v']==V]
    stations = stations[stations['tile_h']==H]
    
    snames = stations['Station'].unique()
    
    return stations, snames


def inverse_mapping(i, j, H, V, xmin = -20015109, ymax = 10007555 , T = 1111950, size = 2400):
    w = T/size
    x = (j + 0.5)*w + H*T + xmin 
    y = ymax - (i + 0.5)*w - V*T
    lat = y/R
    long = x/(R*math.cos(lat))
    
    return lat,long

def deg_to_rad(lat_deg, lon_deg):
    lat_rad = lat_deg * math.pi / 180
    long_rad = lon_deg * math.pi /180
    
    return lat_rad, long_rad

def rad_to_deg(lat_rad, lon_rat):
    lat_deg = (lat_rad * 180) / math.pi
    lon_deg = (lon_rad * 180) / math.pi
    
    return lat_deg, lon_deg

def get_modis_layer(file_name = 'MOD13A1.A2017353.h18v03.006.2018004225447.hdf' , file_path = '/data/co2flux/common/mdomhoef/Oslo/Oslo_analysis/data/MODIS/2018_h18_v03/', layer_name = '500m 16 days EVI'):
    # date = str.split(file_name, sep = ".")[1]
    # d = datetime.datetime.strptime(date[3:], '%y%j').date()
    # title = str(d) + '_' + layer_name
    file_path = file_path + file_name
    file = SD(file_path, SDC.READ)
    sds_obj = file.select(layer_name)
    scale_factor = sds_obj.attributes()['scale_factor']
    data = sds_obj.get() # get sds data
    data = data/scale_factor
    data[data==-3000/scale_factor] = np.nan
    
    return data
    
def forward_mapping(lat, long, R = 6371007.181, xmin = -20015109, ymax = 10007555 , T = 1111950 , size = 2400):
    w = T/size
    x = R * long * math.cos(lat) 
    y = R * lat 

    H = math.floor(((x-xmin) / T))
    # H = (x-xmin) / T
    V = math.floor(((ymax-y)/T))
    # V = (ymax-y)/T

    i = math.floor(((ymax -y)%T)/w -0.5 )
    j = math.floor(((x -xmin)%T)/w -0.5 )
    return i, j

# Building the model using gradient decent to update alpha and beta 
# https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
# Y = mX + c
# Reco = alpha*temp + b 
def lin_reg_grad_dec(m, c, X, Y, L = 0.0001, epochs = 1000):

    '''
    Y = mX + c : Simple linear function 
    L: The learning rate 
    epochs: The number of iterations to perform gradient descent
    
    returns: the slope m and y intercept c 

    '''

    n = float(len(X)) # Number of elements in X

    # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = m*X + c  # The current predicted value of Y
        # D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_m = (-2/n) * (X * (Y - Y_pred)).sum()  # Derivative wrt m
        # D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        D_c = (-2/n) * (Y - Y_pred).sum()  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        
    return m,c 

def reco(alpha, beta, temp):
    return alpha * temp + beta

def GEE_calculate(lambdaGPP, Tscale, Wscale, Pscale, EVI, Rad, radZero):
    GEE = lambdaGPP*Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/radZero))*(-1)
    GEE[GEE > 0] = 0
    # GEE = GEE *3600
    return GEE

def m_rmse(original, prediction):
     return mean_squared_error(original, prediction, squared=False)
    

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# stations, ls = get_stations_year(2018)
# print(stations, ls)

def get_station_observation_year(StationDataPath, sitename, year):
    fls = os.listdir(StationDataPath)
    fls = [x for x, y in zip(fls, [(sitename in file) for file in fls]) if y == True]
    df = pd.read_csv(StationDataPath+fls[0], index_col = 0)
    df.date = pd.to_datetime(df.date)
    df['year'] = df['date'].dt.year
    df = df[df['year'] == year]
    return df

# objective function
def func_reco(parameters, *data):
    alpha, beta = parameters
    temp, reco = data
    result = 0
    # print(alpha,beta, temp, reco)
    for i in range(len(temp)):
        result += ((alpha*temp[i] + beta) - reco[i])**2

    return result**0.5

# objective function
def func_reco_lm(parameters, *data):
    alpha, beta = parameters
    temp, reco = data
    result = (alpha*temp + beta) - reco

    return result
    
# objective function
def func_gee(parameters, *data):
    lambdaGPP, radZero = parameters
    Tscale,Wscale,Pscale,EVI,Rad, GEE_p = data
    result = 0
    # print(alpha,beta, temp, reco)
    for i in range(len(GEE_p)):
        gee = lambdaGPP*Tscale[i]*Wscale[i]*Pscale[i]*EVI[i]*Rad[i]/(1 + (Rad[i]/radZero))*(-1)
        result += (gee - GEE_p[i])**2

    return result**0.5

def func_gee_lm(parameters, *data):
    lambdaGPP, radZero = parameters
    Tscale,Wscale,Pscale,EVI,Rad, GEE_p = data
    gee = lambdaGPP*Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/radZero))*(-1)
    result = gee - GEE_p
    # print(alpha,beta, temp, reco)

    return result

def func_nee(parameters, *data):
    alpha, beta, lambdaGPP, radZero = parameters
    temp,Tscale, Wscale, Pscale, EVI, Rad, nee, = data
    result = 0
    # print(alpha,beta, temp, reco)
    nee = nee/3600
    for i in range(len(temp)):
        p_reco = (alpha*temp[i] + beta)
        p_gee = lambdaGPP*Tscale[i]*Wscale[i]*Pscale[i]*EVI[i]*Rad[i]/(1 + (Rad[i]/radZero))*(-1)
        result += ((p_gee +p_reco)- nee[i])**2

    return result**0.5

def func_nee_lm(parameters, *data):
    alpha, beta, lambdaGPP, radZero = parameters
    temp,Tscale, Wscale, Pscale, EVI, Rad, nee, = data
    result = 0
    # print(alpha,beta, temp, reco)
    p_reco = (alpha*temp + beta)
    p_gee = lambdaGPP*Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/radZero))*(-1)
    result += ((p_gee +p_reco)*3600 - nee)

    return result

def nee_grad_dec(a,b,l,r, temp,Tscale, Wscale, Pscale, EVI, Rad, nee, L = 0.0001, epochs = 1000):
    # temp,Tscale, Wscale, Pscale, EVI, Rad, nee, = data
    n = float(len(nee))
    nee = nee/3600
    
    for i in range(epochs): 
        reco_p = (a*temp + b)
        gpp_p = l*Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/r))*(-1)
        error = ((gpp_p + reco_p) -nee)
        
        X1 = gpp_p + temp
        X2 = gpp_p
        X3 = (Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/r))*(-1)) + reco_p
        X4 = (l*Tscale*Wscale*Pscale*EVI*(Rad**2)/((Rad+r)**2)*(-1)) + reco_p
        
        D_a = (2/n) * (X1 * error).sum()
        D_b = (2/n) * (X2 * error).sum() 
        D_l = (2/n) * (X3 * error).sum()
        D_r = (2/n) * (X4 * error).sum()
        
        a = a - L * D_a
        b = b - L * D_b
        l = l - L * D_l
        r = r - L * D_r
        
    return a,b,l,r
        
    

# gradient decent gpp 
def gpp_grad_dec(m, Tscale, Wscale, Pscale, EVI, Rad, c, Y, L = 0.0001, epochs = 1000):

    '''
    Y = mX + c : Simple linear function 
    L: The learning rate 
    epochs: The number of iterations to perform gradient descent
    
    returns: the slope m and y intercept c 
    '''
    n = float(len(Y)) # Number of elements in X
    # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = GEE_calculate(m, Tscale, Wscale, Pscale, EVI, Rad, c)# The current predicted value of Y
        #When using this mean squared error, get weird grad decent results 
        # error = np.mean(((Y - Y_pred)**2))
        error = Y -Y_pred
        X1 = Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/c))*(-1)
        X2 = m*Tscale*Wscale*Pscale*EVI*(Rad**2)/((Rad+c)**2)*(-1)
        D_m = (-1/n) * sum(X1 * (error))
        D_c = (-1/n) * sum(X2 * (error))
        # D_m = (-2/n) * sum(X1 * (Y - Y_pred))  # Derivative wrt m
        # D_c = (-2/n) * sum(X2 * (Y - Y_pred))  # Derivative wrt m
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        
    return m,c 


def load_data_gpp(params, station_data, sitename, year, StationDataPath, months = None):
        T_low = [4,0,2,3,0,0,0,-999,0]
    
        df = station_data.filter(regex=sitename)
        
        iveg = params.loc[sitename, 'iveg']
        #params morris
        TMIN = params.loc[sitename, 'Tmin']
        TMAX = params.loc[sitename, 'Tmax']
        TOPT = params.loc[sitename, 'Topt']
        tlow = T_low[iveg]
        Temp = df[sitename + '_TEMP_Station']
        EVI = df[sitename + '_EVI']
        LSWI = df[sitename + '_LSWI']
        Rad = df[sitename + '_RAD_Station']
        EVImax = np.nanmax(EVI)
        EVImin = np.nanmin(EVI)
        LSWImax = np.nanmax(LSWI)
        LSWImin = np.nanmin(LSWI)

        Tscale = ((Temp - TMIN)*(Temp-TMAX))/(((Temp-TMIN)*(Temp-TMAX))-((Temp-TOPT)*(Temp-TOPT)))
        Tscale[Tscale < 0] = 0

        if iveg in [3, 6]:
            Wscale = (LSWI - LSWImin)/(LSWImax - LSWImin)
        else:
            Wscale = (1 + LSWI)/(1 + LSWImax)

        Wscale[Wscale < 0] = 0

        Pscale = (1 + LSWI)/2
        if iveg == 0:
            Pscale[:] = 1
        
        # print('iveg: ', iveg)

        if iveg in [1, 2, 3, 5, 7, 8]:
            threshmark = 0.55
            evithresh = EVImin + (threshmark*(EVImax-EVImin))
            # print(EVI,'\n', evithresh)
        
            phenologyselect = np.where(EVI> evithresh)
            Pscale[phenologyselect[0]] = 1
        #by default, grasslands and savannas never have pScale=1 
        Pscale[Pscale < 0] = 0

        lambdaGPP = params.loc[sitename, 'lambdaGPP']
        radZero = params.loc[sitename, 'radZero']
        
        df_obs = get_station_observation_year(StationDataPath, sitename, year) 
        
        if months != None:
            df_obs['month'] = df_obs.date.dt.month
            df_obs = df_obs[df_obs['month']!= 12]
            df_obs = df_obs[df_obs['month']!= 1]
            df_obs = df_obs[df_obs['month']!= 2]
            
        if 'GPP_DT_VUT_REF' in df_obs.columns:
            label = 'GPP_DT_VUT_REF'
        else:
            label = 'FC'

        df_obs.loc[df_obs[label] < -9990, label] = np.nan
        df_obs[label] = df_obs[label]#*3600
        df_obs.set_index('TIMESTAMP_START', inplace=True)
        df_obs = df_obs[[label]]
        GPP_obs = df_obs[label].values
        
        return lambdaGPP, Tscale, Wscale, Pscale, EVI, Rad, radZero, GPP_obs*(-1)
    
def m_NEE(GPP, RECO):
    GPP = GPP.reset_index(drop = True)
    RECO = RECO.reset_index(drop = True)
    if GPP.max()== -0:
        NEE = GPP.add(RECO)
    elif GPP.max() < 0:
        NEE = GPP.add(RECO)
    else:
        NEE= -GPP.add(RECO)
    
    return NEE

# predict NEE #compare to 0riginal 
def predict_NEE(year, StationDataPath, parameters_wrf, iveg, winter = False):
    '''
    In: 
    - year 
    - StationDataPath 
    - params wrf
    - iveg 
    
    out: 
    - df with rmse of all parameters gained with different methods (morris, diff evolution, gradient decent, wrf) 
    compared to original nee and original calculated nee (reco riginal + gee original)
    - csv file with predictions of different methods and original data 
    '''
    station_data = pd.read_csv('Oslo_analysis/Station_evi_lswi_temp_rad_'+str(year)+'.csv', header = 0, index_col=0)
    
    if winter == True: 
        station_data.loc[:,'month'] = pd.to_datetime(station_data.index).month.to_list()
        station_data = station_data[station_data['month']!= 12]
        station_data = station_data[station_data['month']!= 1]
        station_data = station_data[station_data['month']!= 2]
        
        df_morris_0, df_gpp_0, df_reco_0 = load_params_as_df(year, 0, winter = True)
    
    else:
        df_morris_0, df_gpp_0, df_reco_0 = load_params_as_df(year, 0)
          
    mean_morris = df_morris_0.mean()
    mean_gpp = df_gpp_0.mean()
    mean_reco = df_reco_0.mean()
    
    snames = df_morris_0.index
    

    df_mean_nee = pd.DataFrame(index = snames, columns = ['morr_RMSE', 'gd_RMSE','ev_RMSE', 'wrf_RMSE', 'morr_RMSEc', 'gd_RMSEc','ev_RMSEc', 'wrf_RMSEc'])
    df_preds = pd.DataFrame()
    for sitename in snames:
        df = get_station_observation_year(StationDataPath, sitename, year)
        
        if winter ==True:
            df.date = pd.to_datetime(df.date)
            df['month'] = df['date'].dt.month
            df = df[df['month'] != 12]
            df = df[df['month'] != 2]
            df = df[df['month'] != 1]
        #predict GPP 
        lambdaGPP, Tscale, Wscale, Pscale, EVI, Rad, radZero, GPP_or = load_data_gpp(df_morris_0, station_data, sitename, year, StationDataPath, months = True)
        gee_morris = GEE_calculate(mean_morris.lambdaGPP, Tscale, Wscale, Pscale, EVI, Rad, mean_morris.radZero)
        gee_gd = GEE_calculate(mean_gpp.lambdaGPP_gd, Tscale, Wscale, Pscale, EVI, Rad, mean_gpp.radZero_gd)
        gee_ev = GEE_calculate(mean_gpp.lambdaGPP_ev, Tscale, Wscale, Pscale, EVI, Rad, mean_gpp.radZero_ev)
        gee_wrf = GEE_calculate(parameters_wrf[1][0], Tscale, Wscale, Pscale, EVI, Rad, parameters_wrf[1][1])

        # predict RECO
        reco_or = df['RECO_DT_VUT_REF']    
        t = df['TA_F']    
        reco_morris = reco(mean_morris.alpha, mean_morris.beta, t)
        reco_gd = reco(mean_reco.alpha_gd, mean_reco.beta_gd, t) 
        reco_ev = reco(mean_reco.alpha_ev, mean_reco.beta_ev,t)
        reco_wrf = reco(parameters_wrf[1][2], parameters_wrf[1][3], t)

        #predict NEE params   
        nee_morris = m_NEE(gee_morris, reco_morris)
        nee_gd = m_NEE(gee_gd, reco_gd)
        nee_ev = m_NEE(gee_ev, reco_ev)
        nee_wrf = m_NEE(gee_wrf, reco_wrf)   

        #calculate NEE original
        nee_orc = -df.GPP_DT_VUT_REF + df.RECO_DT_VUT_REF
        #original NEE
        nee_or = df['NEE_VUT_REF']

        #comparison to NEE    
        df_mean_nee.loc[sitename]['morr_RMSE'] = m_rmse(nee_or, nee_morris)
        df_mean_nee.loc[sitename]['gd_RMSE'] = m_rmse(nee_or, nee_gd)
        df_mean_nee.loc[sitename]['ev_RMSE'] =m_rmse(nee_or, nee_ev)
        df_mean_nee.loc[sitename]['wrf_RMSE'] = m_rmse(nee_or, nee_wrf) 

        df_mean_nee.loc[sitename]['morr_RMSEc'] = m_rmse(nee_orc, nee_morris)
        df_mean_nee.loc[sitename]['gd_RMSEc'] = m_rmse(nee_orc, nee_gd)
        df_mean_nee.loc[sitename]['ev_RMSEc'] =m_rmse(nee_orc, nee_ev)
        df_mean_nee.loc[sitename]['wrf_RMSEc'] = m_rmse(nee_orc, nee_wrf)   
        
        #save predictions in csv file 
        df_nee_predictions = pd.DataFrame(index = range(0, len(nee_or)))
        df_nee_predictions['Orig_' + sitename] = nee_or.reset_index(drop = True)
        df_nee_predictions['Orig_cal_' + sitename] = nee_orc.reset_index(drop = True)
        df_nee_predictions['Morris_' + sitename] = nee_morris
        df_nee_predictions['Grad_Dec_' + sitename] = nee_gd
        df_nee_predictions['Diff_Evol_' + sitename] = nee_ev
        df_nee_predictions['WRF_' + sitename] =nee_wrf
                        
        df_preds = pd.concat([df_preds, df_nee_predictions], axis=1)        
        df_preds.to_csv('Oslo_analysis/VPRMoutput/Predictions_Observations_'+str(year)+'.csv')
    return df_preds, df_mean_nee

def load_params_as_df(year, iveg, winter = False):
    '''
    returns dfs of parameters of stations of respective years
    '''
    if winter == False:
        df_gpp = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_gd_lambda_radZero'+str(year)+'.csv', index_col=0)
        df_reco = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_gd_alpha_beta'+str(year)+'.csv', index_col=0)
        df_morris = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_morris_'+str(year)+'.csv', header = None)
    
    else:
        df_gpp = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_gd_lambda_radZero_nowinter'+str(year)+'.csv', index_col=0)
        df_reco = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_gd_alpha_beta_nowinter'+str(year)+'.csv', index_col=0)
        df_morris = pd.read_csv('Oslo_analysis/VPRMoutput/best_fit_morris_nowinter_'+str(year)+'.csv', header = None)
        
    col_list = ['lambdaGPP', 'radZero', 'alpha', 'beta', 'Tmin', 'Tmax', 'Topt', 'Station', 'iveg']
    df_morris.columns = col_list
    df_morris.index = df_morris.Station
    # get stations with same PFT
    df_morris_0 = df_morris[df_morris.iveg == 0]
    df_gpp_0 = df_gpp[df_gpp.index.isin(df_morris_0.index)]
    df_reco_0 = df_reco[df_reco.index.isin(df_morris_0.index)]
    
    return df_morris_0, df_gpp_0, df_reco_0

def vprm_all_for_morris(iveg, params, EVI, LSWI, EVImax, EVImin, LSWImax, LSWImin, Temp, Rad):
    """
    This function estimates the mean GPP, RESP and NEE annually and seasonally for a year and a 
    selected ICOS station using the VPRM model.
    Input:
        - params, list of floats with the VPRM parameters [[par01, lambda1,  alpha1, beta1, Tmin1, Tmax1, Topt1],[lambda2, par02, alpha2, beta2, Tmin2, Tmax2, Topt2],...]
        - evi, numpy array of EVI indexes
        - lswi, numpy array of LSWI indexes
        - evimax, float max EVI value in the year
        - evimin, float min EVI value in the year
        - lswimax, float max LSWI value in the year
        - lswimin, float min LSWI value in the year
        - temp, numpy array of temperatures
        - rad, numpy array of surface solar radiation downwards
    Output:
        - List with GPPan, GPPs1, GPPs2, GPPs3, GPPs4, RESPan, RESPs1, RESPs2, RESPs3, RESPs4, NEEan, NEEs1, NEEs2, NEEs3, NEEs4, RMSEan, RMSEs1, RMSEs2, RMSEs3, RMSEs4
    """
    TMIN = params[4]
    TMAX = params[5]
    TOPT = params[6]
    T_low = [4,0,2,3,0,0,0,-999,0]
    tlow = T_low[iveg]
    
    Tscale = ((Temp - TMIN)*(Temp-TMAX))/(((Temp-TMIN)*(Temp-TMAX))-((Temp-TOPT)*(Temp-TOPT)))
    Tscale[Tscale < 0] = 0
    #modification for so-called "xeric systems", comprising shrublands and grasslands
    #these have different dependencies on ground water.
    
    if iveg in [3, 6]:
        Wscale = (LSWI - LSWImin)/(LSWImax - LSWImin)
    else:
        Wscale = (1 + LSWI)/(1 + LSWImax)

    Wscale[Wscale < 0] = 0
    Pscale = (1 + LSWI)/2

    if iveg == 0:
        Pscale[:] = 1
        
    if iveg in [1, 2, 3, 5, 7, 8]:
        threshmark = 0.55
        evithresh = EVImin + (threshmark*(EVImax-EVImin))
        phenologyselect = np.where(EVI[:] > evithresh)
        Pscale[phenologyselect] = 1
    #by default, grasslands and savannas never have pScale=1
    Pscale[Pscale < 0] = 0

    lambdaGPP = params[1]
    radZero = params[0]
    GEE = lambdaGPP*Tscale*Wscale*Pscale*EVI*Rad/(1 + (Rad/radZero))*(-1)
    
    GEE[GEE > 0] = 0
    GEE = GEE *3600
    
    alpha = params[2]
    beta = params[3]
    Temp[Temp<tlow] = tlow
    
    RSP = Temp*alpha + beta
    
    RSP = RSP *3600
    
    NEE = GEE + RSP
    
    # print(NEE)
    
    return GEE, RSP, NEE
