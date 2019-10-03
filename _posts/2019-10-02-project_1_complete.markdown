---
layout: post
title:      "Project 1 COMPLETE!"
date:       2019-10-03 01:33:54 +0000
permalink:  project_1_complete
---


I am still taking it all in, but I am proud to say that I have made my way through my first big data science project! I was a little hesitant at first, but as I worked my way through, my confidence definitely kept growing. As I was going through my finalized project, I started to take note of things that I learned and what aspects of the project were surprising.  One of the aspects that stood out to me as not being quite what I expected was the process of cleaning and scrubbing the data before even beginning to analyze the data. I knew that no dataset comes all tied up in a bow and ready to go, but I was a bit surprised at how many different steps need to be taken to get the data ready to be investigated.

For my project, I took 5 different steps to scrub my data, all of which were important actions to remove any aspects that could be harmful to the analysis.  First, I changed any necessary variables to the appropriate numerical format, as only numerical data can be used in a regression analysis. Second, I replaced any null values, as null values hinder our ability to build a model. Third, I investigated potential multicollinearity between variables, which if present can give false impressions of the impact of certain variables. Fourth, I removed any columns that seemed unnecessary to include. Fifth and final, I identified any columns that should be categorical and dealt with them appropriately.  That’s a lot to be done before really even getting to start to look at the data in a meaningful way!  

Two of these steps, step three and five, were areas that I learned the true importance of during this project, and that I would like to share my thoughts/process on in more depth for anyone else who is as unfamiliar with these as I was prior to this project.


### **Multicollinearity**


This is something that I was not very familiar with, but I am glad I got the opportunity to explore it a bit more with this analysis. Multicollinearity is a phenomenon in a multiple regression analysis when two variables are highly linearly related. With a linear regression model, we are trying to determine the effect of changing a variable while keeping all other variables constant.  If we have two variables that are highly correlated (linearly related), then when we go to change variable #1, we will not know if the impact on our model is a result of the change in variable #1 or if it’s from the indirect change in variable #2. This could lead to false impressions of the importance that each variable has.  Therefore, it is important to take a look at the correlations between each of our variables to every other variable so that we can determine if there are any multicollinearity concerns.  If there are highly correlated variables, then one of them will need to be removed. 

To investigate multicollinearity, I discovered a few different ways to look at a correlation matrix This matrix shows you the correlation between each set of 2 variables. You can look at the raw correlation numbers (which will be between -1 and 1), the absolute values compared to see if a threshold is met (correlation of .7-.8+ is considered high) or a heatmap of the correlations for a visual option. In my case, I had just one set of 2 variables that were highly correlated, the square footage of the house and the square footage of the upstairs.  This means that if I were looking at my model and saw that the price went up when I increased the square footage of the upstairs area, I would not know for sure if the price increase was truly due to the upstairs area or if it was indirectly from the corresponding increase in square footage of the whole house, which is correlated with the upstairs area square footage. To eliminate this problem, I removed the square footage of the upstairs column since that is included within the square footage of the house variable.

Code used:

```
# correlation numbers:
kch_data.corr()
# Absolute value of correlations:
Abs(kch_data.corr()) > .75
# Heatmap of correlations:
sns.heatmap(kch_data.corr(), center=0, linewidths=.5, cmap="Blues");
```


### **Dealing With Categorical Data**

My final step of data scrubbing was to identify which columns should be categorical. For a regression model, we prefer to have continuous data rather than categorical data.  The obvious categorical data is included as text rather than numbers. However, numbered data can also be categorical, and so some investigating needs to be done do determine which numerical columns truly are numerical, and which should be categorical. While this all makes logical sense, I had never really thought about numerical data being considered categorical data before, so I was intrigued when I started to investigate this a bit more. 

I decided to use scatterplots of each variable to determine its true type value. I find the visuals can be easier to interpret since it’s easier to see patterns emerge. Scatterplots that have clear linear buckets of data rather than randomly scattered data indicate to me that that variable should be categorical data.  For example, one of the features in this dataset tells us if the house has a waterfront view or not. There are only two options here, the house can have a view (represented by ‘1’ in the dataset), or it could not have a view (represented by ‘0’ in the dataset). Each of these two options could be considered their own category, and therefore this should be a categorical column even though it originally had numerical values of ‘0’ or ‘1’. When looking at a scatterplot of this waterfront variable, it is pretty easy to see that all of the values fall either into the ‘0’ or ‘1’ bucket. 

Once identified, it is important to transform this data so that we can still utilize it for our analysis. For this project, I first changed the datatype to ‘category’. I then did one hot encoding with dummy variables. This is a method that converts each category into a new column and assigns either a 1 or 0 to the data in the new column. For example, if we are looking at the data for the number of floors in a house, we will have data consisting of say 1 floor, 2 floors, and 3 floors. If we were to do one hot encoding on this data, we would separate the data into 3 columns (1 floor, 2 floors, 3 floors) and each would be filled with zeros for data points that do not have that number of floors and have ones for data points that do have the respective number of floors.  One hot encoding is definitely an interesting method that allows us to actually utilize our categorical data in a meaningful way, rather than having to ignore any categorical data.

Code used:

```
# Scatterplot of each variable with price
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16,20))
for i, xcol in enumerate(['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
                         'view','condition','grade','sqft_basement','yr_built','yr_renovated',
                         'zipcode','lat','long','sqft_living15','sqft_lot15']):
    kch_data.plot(kind='scatter', x=xcol, y='price', ax=axes[i//4][i % 4], alpha=0.4, color='b')
	
# One hot encoding for the waterfront variable

kch_data.waterfront = kch_data.waterfront.astype('category')
waterfront_dummies = pd.get_dummies(kch_data.waterfront, prefix='waterfront', drop_first=True)
kch_data = kch_data.drop(["waterfront"], axis=1)
kch_data = pd.concat([,waterfront_dummies], axis=1)
```

