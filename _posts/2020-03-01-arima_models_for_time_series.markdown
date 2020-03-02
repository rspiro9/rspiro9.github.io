---
layout: post
title:      "ARIMA Models For Time Series"
date:       2020-03-02 02:38:18 +0000
permalink:  arima_models_for_time_series
---


Time series datasets have the progress of time as a main dimension in the data. In other words, we are recording data over set intervals of time. One of the benefits of having time series data is that you can use it to forecast future values. In this post, I will be walking through how to forecast a time series dataset using an ARIMA model. The data I am working with includes average house prices per county in the US over a 22 year period (from Zillow).

The first thing to do is get the data into the right format. In order for the model to properly read our dates, we need them to be in datetime format. Pandas has a convenient to_datetime function that allows us to easily do so.  I created a quick function that can be easily re-used to switch the dates into the proper format. With my dataset, the columns including date-related data start in column 7.

```
def get_datetimes(df):
   return pd.to_datetime(df.columns.values[7:], format='%Y-%m')
```
		
		
After having my data in this format, I preprocessed the data (i.e. removed null values) and did some exploratory analysis to get a better understanding of the data. For the purposes of this post though, I am not going to dive into any of those details.

In the above dataframe, our dates are actually each in their own column, which is known as wide format. However, for the ARIMA analysis, we need the dates to all be in one column, known as long format. I chose to create a function that would take a list of regions and transform the data into the proper long format, with each regions' data in its own column. Here is the function that will perform our dataframe reformating:

```
def regions_dataframe(regions, start = '2012-04-01', end = '2018-04-01', data = data):
    
    '''
    Given a list of regions along with a start and end date, this returns a long format time series dataframe 
    with each region having its own column
    
    Parameters:
    - regions (list) - The regions we want to keep
    - start (str) - The date we want to start with in our dataframe ('YYYY-MM-DD')
    - end (str): The date to we want to end with in our dataframe ('YYYY-MM-DD')
    - data (dataframe): The dataframe we will use
    '''
    
    # If a single region is entered, turn this entry into a list:
    if type(regions) != list:
        regions = [regions]
    
    # Update the start and end date to be in datetime format:
    start = pd.to_datetime(start, format='%Y-%m')
    end = pd.to_datetime(end, format='%Y-%m')
    
    # Create a new dataframe that only includes the regions we want to look at:
    data_temp = data.loc[data['RegionName'].isin(regions)]
    
    # Set the RegionName column to be in string format (needed for ARIMA model) and reset this column to be the index:
    data_temp['RegionName'] = data_temp['RegionName'].astype(str)
    data_temp.set_index(['RegionName'],inplace=True)

    # Identify the columns representing our start date and our end date:
    columns = list(data_temp.columns)
    start_index = columns.index(start)
    end_index = columns.index(end)
    
    # Isolate the columns within our start and end dates. Then transpose the dataframe so the dates are in the index:
    data_temp = data_temp.iloc[:,start_index:end_index+1].T
    
    # Reset the index back to the date.
    data_temp.reset_index(inplace=True)
    data_temp.set_index('index',inplace=True)
    
    return data_temp
```

Now that we have the data in the format we need, let's dive into the ARIMA model.

ARIMA (Auto Regressive Integrated Moving Average) models a regression based on past values, which can be used to forecast future values. ARIMA models are used when the dataset is not stationary (meaning its statistical values, such as mean or variance, do not stay constant over time) and does not have seasonality (meaning trends appear in a cyclical manner). Our dataset meets both of these, which is why we are using the ARIMA model.  The ARIMA model consists of 3 different parts; auto-regressive(AR), intigration (I), moving Average (MA). Let's breakdown these 3 main components.

1. **AR (Auto-Regressive)** - for this part, a value from a time series is regressed on previous values from the same time series. The parameter used to represent this part is 'p', which incorporates the effect of past values into our model.
2. **I (Integration)** - our dataset does not have stationarity, but for a time series to be properly modeled, we actually need to have stationarity. Therefore, the ARIMA model has an integrated component to fix stationarity through differencing. The parameter used to represent this part is 'd', which is the integrated component that deals with trends. This identifies the amount of differencing needed based on the number of lag values to subtract from the current observation.
3. **MA (Moving Average)** - this is a weighted sum of today's and yesterday's noise. the parameter used in ARIMA is 'q' which is the moving average used to set the error of the model as a linear combination of the error values observed at previous time points.

When running our ARIMA model, we will cycle through multiple different combinations of p, d, and q values in order to see which combination creates the best possible model.  To create all of these combinations, use the below code:

```
p = q = d = [0,1,2]
combs = list(itertools.product(p,d,q))
```

Once we have our parameter options identified, it is time to run the ARIMA model and select the best model. There are a few different ways to identify which is the best model. With my dataset, I have decided to use the root mean squared error between the predicted training data and the test data. The model with the lowest root mean squared error will be chosen as the best model since it indicates greater accuracy of our prediction compared to the real data. Below is the function I created to select a model.

```
def arima_selection(dataset, combs=combs, split=0.7):
    
    '''
    Given a dataset, this function will provide us with the best (p,d,q) order that minimizes the root mean squared
    error on the test set and the best fitted ARIMA model
    
    Parameters:
    - dataset -  The dataset we will be fitting to the model
    - combs - List of the possible combinations of (p,d,q)
    - split - What percent of the data we want in our training set
    '''
    
    # Calculate the index to be used to define the train test split:
    limit = int(len(dataset) * split)
    
    # Create the train and test sets:
    train = dataset[:limit]
    test = dataset[limit:]
    
    # Declare each variable that we will work with (to be updated later):
    rmse = None
    best_model = None
    best_order = None
    best_rmse = 10000000
    
    # Loop through all (p,d,q) combinations:
    for i in combs:
        try:
            # Run ARIMA model and fit it to the training set
            model = ARIMA(train,order=i)
            output = model.fit()
            # Forecast the same length as the length of the testing set:
            pred = output.forecast(len(test))[0]
            # Calculate the root mean square error:
            rmse = mean_squared_error(test, pred)**0.5
            
            # If there is a new best RMSE, update the RMSE and model parameter variables:
            if rmse < best_rmse:
                best_order = i
                best_rmse = rmse
                best_model = output

        except:
            continue
            
    if rmse == None:
        return None
    else:
        return best_order, best_model
```

Once we have our best model selected, it is time to forecast future values. In addition to the forecasted value, we also want to capture the upper and lower bounds of the prediction interval, which will help determine how accurate the forecasted value is and what the potential range of this value could be. The below function can be used to forecast future values and provide the lower and upper bounds of the prediction interval:

```
def arima_forecast(output, periods):
    
    '''
    Given output from a fitted ARIMA model and a designated number of periods, provides a series of forecasts, 
    lower bounds, and upper bounds.
    
    Parameters:
    - output - A fitted ARIMA model (in object format)
    - periods - # of periods into the future we want to forecast
    '''

    # Calculate the forecast, lower bounds, and upper bounds:
    forecast = output.forecast(periods)
    lower = [i[0] for i in forecast[2]]
    upper = [i[1] for i in forecast[2]]

    return forecast[0], lower, upper
```

Great, now we are able to use this function to determine what the forecasted values are! Depending on what you are trying to determine, you may be looking for different things from this outcome. For example, you may be more interested in the highest forecasted values, or you may instead be interested in having the narrowest prediction interval bounds. Once you have determine which pieces of the data you want to work with (in our case, we wanted the best 5 regions), plotting the forecasted values along with their bounds is very helpful. Below is a function to do so:

```
def plot_forecasts(regions, labels=None, plot_width=True,
                   start='2012-04', months=120, size=(16,10)):    
    '''
    Given a list of regions, plot their actual values for a desired number of periods along with their 
    future predicted values for a desired number of periods
    
    Parameters:
    - regions - list of region(s) to plot 
    - labels - region names for the legend
    - plot_width - if True plots the width of the forecast
    - start - first date we want to include
    - months - # of forecasted months to include
    - size - size of the plot
    '''
    
    # If single region provided, convert it to a list:
    if type(regions) != list:
        regions = [regions]
    
    # Create a plot figure:
    plt.figure(figsize=size)
    
    # Create a color counter:
    counter=0
    
    # Iterate through each region and plot the desired values:
    for i in regions:
        # Select the real values from past years:
        data_real = regions_dataframe(i).iloc[:-7,:]
        x_real = list(data_real.index)
        y_real = data_real

        # Indicate how long the x axis needs to be for the forecasts:
        x_length = pd.date_range('2017-10-01', periods=months, freq='MS')
        
        # Identify the desired region:
        region_i = arima_data.loc[arima_data['RegionName']==i]
        
        # Specify the label to be associated with that region:
        if labels == None:
            label = f"{region_i['County'].item()} County, {region_i['State'].item()} ({region_i['RegionName'].item()})"
        else:
            label = f"{labels[counter]}, {region_i['State'].item()}"
        
        # Select the region's forecasted values:
        y_forecast = region_i['Forecast'].item()[:months]

        # Identify the color to be used for each region:
        color = color_list[counter]
        
        # Plot the real and forecasted values:
        plt.plot(x_real, y_real, color=color, label = label, lw = 4)
        plt.plot(x_length,y_forecast, color=color, ls='--', label = '', lw = 4)
        
        # If showing the interval widths, plot the bounds and shade between them:
        if plot_width==True:
            lower = region_i['Lower'].item()[:months]
            upper = region_i['Upper'].item()[:months]
            plt.fill_between(x_length, lower, upper,
                 facecolor=color, alpha = 0.15, interpolate=True)
        
        # Add 1 to the counter so a new color is chosen for the next region:
        counter += 1

    # Plot labels and axis limits:
    plt.title('Housing Price Forecasts and Bounds')
    plt.xlabel('Year')
    plt.ylabel('Avearge House Prices in Region ($)');
    
    plt.xlim(pd.to_datetime('2012-04', format='%Y-%m'),x_length[-1])
    plt.ylim(0,)

    plt.legend(loc=2, fontsize=14, frameon=True);
```
