---
layout: post
title:      "NYC Restaurant Yelp and Inspection Analysis"
date:       2020-11-27 11:33:18 -0500
permalink:  nyc_restaurant_yelp_and_inspection_analysis
---


**Background**

New York City is known as one of the best food cities in the world. It has a diverse selection of cuisines and restaurant types, so you can truly find pretty much any type of food you could imagine. With so many unique types of restaurants and the ability to be located in such a populous, food forward city, there is a lot of opportunity for NYC restaurants to be very successul. One of the key factors that impacts the potential success of a restaurant is how many customers go in and order a meal.  With today's technology forward world, many people turn to online research to help them pick which restaurant they should or should not go to. Yelp is often a starting research point for many as it allows consumers to get an idea of other consumers' experience at each restaurant. Having a high Yelp rating is likely to draw in consumers, while a low rating could scare consumers away. Another important aspect that many consumers consider prior to eating at a restaurant is what inspection grade the restaurant has been given. Receiving a 'B' or 'C' grade could indicate that significant sanitary violations were found at the restaurant, which likely will deter many from eating there.

For this analysis, we will be diving into Yelp restauran data and inspection grade data for NYC restaurants, specifically in Manhattan. This analysis will be most helpful for people in or interested in joining the restaurant world in NYC as it can help them identify certain aspects (i.e. cuisine type, location, price level, etc.) that typically lead to higher Yelp ratings or inspection grades, and therefore should be optimized, or lower Yelp ratings or inspection grades, and therefore may want to be avoided.


## Obtaining Data

I have used two different datasets that I merged together for our final analysis. The datasets are:
1. **Yelp Data** - Collected from the [Yelp](https://api.yelp.com/v3/businesses/search) website using the Yelp API on 8/25/20.  This dataset contains information and rating/review details about restaurants from Manhattan. 
2. **Inspection Grade Data** - Provides information about restaurant inspections for all NYC restaurants. The data was collected from the [NYC Open Data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) website on 8/25/20.  This dataset includes restaurants from all 5 NYC buroughs, but for our main analyses we are just leveraging data for Manhattan restaurants.

For the inspection grade data, I downloaded a csv of the data that I then uploaded to my workbook.

For the Yelp data, the process to compile our data was a bit complex due to the limitations of only being able to collect 1,000 rows ff data at a time. However, we wanted to be able to collect data on all Manhattan restaurants, and there are many more than 1,000 restaurants.  Therefore, I used a workaround to collect data on 1,000 restaurants from each Manhattan neighborhood. The list of Manhattan neighborhoods used can be found [here](https://www.nyctourist.com/million-manhattan.htm). 

### Scrubbing Yelp data

I begin the same way I would to scrub any API data, with getting our keys and drawing out the API key, pulling the URL, and identifying the headers. 
```
# Define function to gather keys:
def get_keys(path):
    with open(path) as f:
        return json.load(f)
				
# Pull in keys and specifically draw out the api key. I have removed the specific path to the keys 
# for security purposes:
keys = get_keys("/Users/'INSERTPATH'/yelp_api.json")
api_key = keys['api_key']

# URL to pull data from:
url = 'https://api.yelp.com/v3/businesses/search'

# Identify headers:
headers = {'Authorization': 'Bearer {}'.format(api_key)}
```
 Next, I listed out each of the Manhattan neighborhoods to pull data from (list of neighborhoods used can be found [here](https://www.nyctourist.com/million-manhattan.htm) and created a temporary dataframe that would hold our data.
 ```
 # List of Manhattan neighborhoods:
neighborhoods = ['Midtown West', 'Greenwich Village', 'East Harlem', 'Upper East Side', 'Midtown East',
                 'Gramercy', 'Little Italy', 'Chinatown', 'SoHo', 'Harlem',
                 'Upper West Side', 'Tribeca', 'Garment District', 'Stuyvesant Town', 'Financial District',
                 'Chelsea', 'Morningside Heights', 'Times Square', 'Murray Hill', 'East Village',
                 'Lower East Side', 'Hell\s Kitchen', 'Central Park']
# Create temporary dataframe to hold data:
nyc = [[] for i in range(len(neighborhoods))] 
```
I then created a function that would loop through the list of neighborhoods and pull in 1,000 rows of data from Yelp for each neighborhood. The data was stored in the 'nyc' temporary dataframe we just created.
```
# Function to draw in data for each neighborhood:
for x in range(len(neighborhoods)):
    print('---------------------------------------------')
    print('Gathering Data for {}'.format(neighborhoods[x]))
    print('---------------------------------------------')


    for y in range(20):
        location = neighborhoods[x]+', Manhattan, NY'
        term = "Restaurants"
        search_limit = 50
        offset = 50 * y
        categories = "(restaurants, All)"
        sort_by = 'distance'

        url_params = {
                        'location': location.replace(' ', '+'),
                        'term' : term,
                        'limit': search_limit,
                        'offset': offset,
                        'categories': categories,
                        'sorty_by': sort_by
                    }
        
        response = requests.get(url, headers=headers, params=url_params)
        print('***** {} Restaurants #{} - #{} ....{}'.format(neighborhoods[x], 
                                                             offset+1, offset+search_limit,
                                                             response))
        nyc[x].append(response)

print(response)
print(type(response.text))
print(response.json().keys())
print(response.text[:1000])
```
 Once I had all of the data pulled, I then had to check if there were any blank rows of data since the function pulled in 1,000 rows for each neighborhood, but some neighborhoods may not have 1,000 restaurants. Since we pulled our data 50 rows at a time and repeated that process 20 times to get to our 1,000 restaurants, we will check if any of the 20 groups have less than 50 rows of data in them. 
```
## Check for any empty business lists:
for x in range(len(neighborhoods)):
    for y in range(20):
        num = len(nyc[x][y].json()['businesses'])
        if num != 50:
            print(neighborhoods[x], y, num)
```
We ended up with two neighborhoods that had fewer than 1,000 restaurants: Little Italy and Stuyvesant Town. We therefore will create a new dataframe with only the rows that have true data in them (the rows needed in LittleItaly and Stuyvesant Town were identified in the previous step).

```
## Save the compiled data into dataframe and remove any empty data:
df = pd.DataFrame()
for x in range(len(neighborhoods)):
    if x == 6: # Little Italy has a total of 486 restaurants
        for y in range(10):
            df_temp = pd.DataFrame.from_dict(nyc[x][y].json()['businesses'])
            df_temp.loc[:,'neighborhood'] = neighborhoods[x]
            df = df.append(df_temp)
    if x == 13: # Stuyvesant Town has a total of 417 restaurants
        for y in range(9):
            df_temp = pd.DataFrame.from_dict(nyx[x][y].json()['businesses'])
            df_temp.loc[:,'neighborhood'] = neighborhoods[x]
            df = df.append(df_temp)

    else:
        for y in range(20):
			 	    df_temp = pd.DataFrame.from_dict(nyc[x][y].json()['businesses'])
            df_temp.loc[:,'neighborhood'] = neighborhoods[x]
            df = df.append(df_temp)
```
 
 And with that, we have all of our data!
 
## Scrubbing Data
It is important to scrub your data to make sure that it is optimal for using in your model. For this blog post, I am focusing more on the analysis, so I am not going to go into a lot of details about the scrubbing down. For a more detailed look, you can reference the GitHub link to the full project at the bottom of this post. Here is a brief overview of the scrubbing steps that were taken:
1. Remove duplicate data - some restaurants were listed multiple times in the dataset. Since we only want to look at one entry per restaurant, duplicative rows were removed.
2. Remove unnecessary columns - any columns that were not going to be relevant to our analysis were removed.
3. Feature Engineering - in the Yelp dataset, a few columns (i.e. categories, coordinates, location) consisted of data in dictionary format. Therefore, I extracted these values so that they woould be easeier to work with. Additionally, I used the frequency of each cuisine type to identify which restaurants are mainstream cuisines (cuisine appeared at least 100 times in the dataset) and which are rare cuisines. The PHONE column in the inspection dataset was manipulated to match the format of the display_phone column in the Yelp datset so that this column could be used to merge the datasets later on.
4. Dealing with categorical data - In order to run our analysis, we will need our data to be numerical. Therefore, we used dummy variables to turn the cuisine type, transactions, GRADE, neighborhood, critical_flag, count_range, and price_value columns into numerical data columns.
5. Dealing with missing data - any missing price values were updated to be zero to represent the price level is unknown. The ACTION, CRITICAL FLAG, INSPECTION TYPE, GRADE, SCORE, and GRADE DATE columns were manipulated so that any null values were replaced with values based on indications from the dataset creaters. All other rows with null values were removed.
6. Dealing with outliers - restaurants with a very large number of reviews (greater than 1,000) were updated to say they have 1,000 views to remove outliers but still be able to use the data. Restaurants with a large number of inspections (greater than 70) had their number of inspection visits value replaced with '70' so that outliers were no longer present but the data could still be used and would still indicate a large number of inspections for each particular restaurant updated.

## Exploratory Data Analysis

Once all of the data is scrubbed, it's time to move on to exploring the data and getting to know it better. There were a few key data cuts that I wanted to look at. First, I wanted to investigate the different types of cuisine and see if certain cuisines typically receive higher Yelp grades and/or a greater number of Yelp reviews. To do this, I looked at two different charts. The first is a bar chart showing the percent of restaurant for each cuisine type that received each different Yelp rating (ratings are on a scale of 0-5).
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/%25%20Rating%20Per%20Cuisine.png?raw=true)

From this graph you can see that the majority of cuisines have mostly 4.0 ratings, followed by 3.5. In particular, Korean, Seafood, Pizza/Italian, Japanese, and Asian have the most 4.0 ratings. Cafe/Coffee/Tea and Other appear to have the most 5.0 ratings. Mexican, Pizza, Chinese, and Latin appear to have the most lower Yelp ratings. Next, looking at the number of reviews, we see that on average, Korean restaurants have the most reviews, followed by Seafood and then French food.
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Number%20Reviews%20Per%20Cuisine.png?raw=truehttp://)

Next I took a look at the data by neighborhood. When looking at how Yelp ratings vary by neighborhood, it looks like most neighborhoods have a similar spread of ratings averaging between 3.5 and 4.0, though Morningside Heights appears to typically have lower ratings. 
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Yelp%20Review%20Per%20Neighborhood.png?raw=truehttp://)

Looking at the Inspection Grades, all of the neighborhoods have a majority of 'A' inspection grades. Little Italy has the greatest percentage of 'B' grades and Hell's Kitchen has the greatest percentage of 'C' grades.
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/%25%20Grade%20Per%20Neighborhood.png?raw=truehttp://)

I also investigated the different price levels. In relation to the Inspection Grade, restaurants with a high price level ($$$$) tend to have fewer B and C grades then restaurants with a lower price level ($ or $$ or $$$)
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Grade%20Per%20Price%20Level.png?raw=truehttp://)

Having a higher price ($$$$) also tends to lead to receiving a better Yelp review.
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Prices%20Per%20Rating.png?raw=truehttp://)

I also took a look at the Inspection Grade in relation to both the Yelp Rating and the number of Yelp Reviews. From the below chart it looks like the greatest number of reviews are typically given to restaurants in the 3.5-4.5 range. Few reviews are given for 5 star or 2 and below stars, though this is likely because few restaurants acheived these ratings. Additionally, restaurants with a 2.5 rating and a 'C' inspection grade seem to comparatively have a lot of reviews. This is likely because consumers had a bad experienceat the restaurant and want to share their bad experience with others to warn others about the restaurant. We also see some outliers specifically for restaurants with an 'A' grade, indicating that very positive expereince may help encourage consumers to write reviews.
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Grade%20By%20Number%20Reviews%20and%20Rating.png?raw=truehttp://)

Lastly, I pulled together an interactive map that shows the location of each restaurant. If you hover over a restaurant's marker,  it will tell you the name, rating, price level, and inspection grade of the restaurant. The below code was used to create this map.
```
# Create map:
yelp_map = folium.Map([40.774371, -73.95931], zoom_start=12)

# Select values to be used to identify the location of each restaurant and turn these values into a list:
locations = df_merged[['latitude', 'longitude']]
locationlist = locations.values.tolist()

# Add markers to map and add details (i.e. name, price, rating, grade) to each marker:
for i in range(0,len(locationlist)//2):
    details = "{} \nPrice: {} Rating: {} Grade: {}".format(df_merged['name'][i], str(df_merged['price'][i]),
                                                                           str(df_merged['rating'][i]),
                                                                               str(df_merged['GRADE'][i]))
    popup = folium.Popup(details, parse_html=True)
    folium.Marker(locationlist[i], popup=popup).add_to(yelp_map)

# Show map:
yelp_map3
```
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Map.png?raw=truehttp://)


## Hypothesis Testing
Once I finished up exploring my data, it was time to start analyzing my data. I did two separate analyses: hypothesis testing and machine learning. I started with hypothesis testing.  When doing an analysis, hypothesis testing is often used in order to determine whether or not an outcome is statistically significant. When running a hypothesis test, we must first set our null and alternative hypotheses. The **null hypothesis** is typically that there is no relationship between A and B, with thile **alternative hypothesis** is your educated guess about the outcome (i.e. A is greater than B). To determine weather we reject or accept our null hypothesis, we look at the relationship between the p-value, which is the probability of observing a test statistic at least as large as the one observed, and the alpha value (𝛼), which is the threshold at which we are ok rejecting the null hypothesis. For this analysis I used an alpha value of 𝛼=.05, meaning that if our p-value is less than our alpha value of .05 then we can reject the null hypothesis. If we do end up rejecting our null hypothesis, we can then look at the Effect Size to determine the difference between the observed groups.

I evaluated four questions using hypothesis testing.  The first questions was:
*Does a restaurant's Yelp rating influence how many Yelp reviews the restaurant will receive?*

I began by grouping the data to identify high Yelp ratings (4+) and low Yelp ratings (3.5 and below). I then took a look at the mean scores for the number of reviews for high Yelp rating vs. low Yelp rating to get a general idea of if the means look similar or different.

```
# Update rating to be grouped by high (4+) and low (3.5 and below) ratings:
score = {1: 0, 1.5: 0, 2:0, 2.5: 0, 3: 0, 3.5:0,4:1,4.5:1,5:1}
df_merged['rating_score'] = df_merged['rating'].map(score)

# Select data needed for analysis:
high_star = df_merged[df_merged['rating_score']==1]['review_count']
low_star = df_merged[df_merged['rating_score']==0]['review_count']

# Compare mean scores of number of review for high Yelp ratings vs. low Yelp ratings:
print('High Score Mean:',high_star.mean())
print('Low Score Mean:',low_star.mean())
```
This output mean scores of:
High Score Mean: 352.6193628465039
Low Score Mean: 314.0522141440846

Based on the mean values, it does seem like having a high rating may indicate more reviews have been written for a restaurant. Next I had to choose the appropriate testing method. Typically a t-test is used for hypothesis testing, which tells if there is a statistical difference between the means of two populations. If the sample sizes and/or sample variances of the two groups being tested are equal, then a standard student's t-test would be used. However, if sample size and variances are unequal between the 2 populations, then an adaption of the student's t-test known as a Welch's t-test would be used. I compared the variances and sample sizes using the follow code and found that both were not equal, indicating a Welche's t-test should be used.

```
# Test whether variances and sample size are equal:
print('Are variances equal?:',np.var(high_star) == np.var(low_star))
print('Are sample sizes equal?:',len(high_star) == len(low_star))
```

After this I ran my Welch's t-test. I used a 1-tailed t-test since I was just looking to see if a high rating leads to a greater number of reviews. To run the t-test, I used the ttest_ind function to determine if there is any difference in the number of reviews, and passed the 'equal_var = False' function to indicate that our variances are unequal.

```
# Run 1-sided Welch's t-test:
result = stats.ttest_ind(high_star, low_star, equal_var = False) # 1-tailed Welch's t-test
print('Reject Null Hypothesis' if result[1]/2<.05 else print('Failed to Reject Null Hypothesis'))
print('t-statistic:',result[0],'p-value:',result[1]/2)
```

This resulted in us rejecting the null hypothesis as our p-value of 4.302440410486111e-05 is less than alpha=.05. With this we can say that having a high Yelp rating does lead to a greater number of reviews being written compared to restaurants with low ratings. Once I knew that I would be rejected the null hypothesis, the next step was to determine the size of the difference between the two population means by looking into the **Effect Size**.  Effect size will help us understand the practical significance of our results. In other words, how meaningful is the statistical difference between our two groups. To understand the effect size, I will use Cohen's d, which represents the magnitude of differences between 2 groups on a given variable. Larger values for Cohen's d will indicate greater differentiation between the two groups. A Cohen's d effect size around .2 is considered 'small', around .5 is considered 'medium, and around .8 is considered 'large'.

The formula for Cohen's d is: 𝑑 = effect size (difference of means) / pooled standard deviation. In code form it is:

```
# Cohen's d formula:
def Cohen_d(group1, group2):
    '''This function takes in two groups of data and calculates the Cohen's d value between them.'''

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d
```

In this case, a low Cohen d value of .125 was seen, so we can say having a high Yelp rating has only a small effect on the number of reviews.  

The other three questions that I used hypothesis testing to evaluate along with their outcomes were:

* Does a restaurant's inspection grade influence how many Yelp reviews the restaurant will receive?
*     Reject Null Hypothesis, indicating having a higher inspection grade does lead to having a higher number of Yelp reviews received, though the effect size is small.
* Does the type of cuisine influence how many Yelp reviews the restaurant will receive?
*     Reject Null Hypothesis, indicating some cuisine types do impact the number of Yelp reviews received.
* Is there a relationship between the Inspection Grade and the Neighborhood, Price, or Cuisine Type?
*     Reject Null Hypothesis, as some price levels can have an impact on the inspection grade, such as when we look at 1 dollar sign vs. 2 dollar signs or 2 dollar signs vs. 4 dollar signs. However, we fail to reject the null hypothesis when looking specifically at the impact of the neighborhood or the cuisine type on inspection grade.


## Machine Learning (Classification)
Machine Learning is a way to run data analyses by using automated analytical models that have the capability to learn. For this analysis, I have used classification, a type of supervised machine learning that uses labelled data to help predict which group a data point should be classified into. I ran 5 different classification models in order to determine which of the five would be most accurate and therefore should be utilized. The five classification models used are:

Decision Tree
Randoom Forest
Adaboost
XGBoost
Logistic Regression

The first step was to identify the target data and then split the data into train and test sets. This allows us to ensure that we are not overfitting or underfitting our data too much.

```
# Convert ratings columns values so that the target variable will not be continuous: 
df_merged['high_rating'] = df_merged['rating'].apply(lambda x: 1 if x > 3.5 else 0)

# Identify our X and y variables:
X = df_merged.drop(columns=['CAMIS','DBA','BORO','BUILDING','STREET','ZIPCODE','display_phone','CUISINE DESCRIPTION',
                           'INSPECTION DATE','ACTION','SCORE','GRADE DATE','INSPECTION TYPE','Latitude','Longitude',
                           '#_of_inspections','name','price','rating','review_count','transactions',
                            'categories_clean','latitude','longitude','address','city','zip_code',
                            'state','num_of_cat','high_rating',],
                   axis=1)
y = df_merged.loc[:,'high_rating']

# Split into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```

A few of the data columns had very few data points in them. Once they were split into training and testing data, we ended up with certain columns having all of their data in one of the new train/test datasets and no values in the other dataset. Therefore, I used the below function to remove all of the columns that have no data in them. I removed these columns in both the training and testing datasets to ensure I still had the same columns present in each set.

```
# Create function that identifies columns with no entries:
def remove_columns(data):
    '''This function will take a dataset and identify any columns whose values sum to zero,
    indicating no data is present in the column.'''
    
    e_dataframe = []
    temp_dataframe = []
    columns = list(data.columns.values)

    for i in columns:
        if data[i].sum().any() == 0:
            e_dataframe.append(i)
        else:
            temp_dataframe.append(i)
    
    return e_dataframe

# Run function to identify columns that need to be removed for both X_train and X_test datasets
empty1 = remove_columns(X_train)
empty2 = remove_columns(X_test)

# Merge the two lists of columns that need to be removed. Use set() function to remove duplicates:
empty = list(set(empty1 + empty2))

# Drop columns that are included in the 'empty' list
X_train = X_train.drop(columns=empty,axis=1)
X_test = X_test.drop(columns=empty,axis=1)
X = X.drop(columns=empty,axis=1)

# Ensure our X_train and X_test sets have the same number of columns:
print('X_train # of columns:', len(X_train.columns),'\n','X_test # of columns:',len(X_test.columns),'\n',
      'X # of columns:',len(X.columns))
```

For my analysis, the results I looked it consisted of a classification report and a confusion matrix. The classification report provides four key metrics in evaluating the performance of a model. The four metrics are:

* **Precision** - measures how accurate the predictions are
*       Precision = TP / (TP + FP)
* **Recall** - % of positives correctly identified
*       Recall = TP / (TP + FN)
* **F1-score** - Harmonic mean of precision and recall.
*       F1-score = 2 x (precision x recall)/(precision + recall)
* **Support** - # of samples of the true responses that are in the class.

A good visual way to visualize how accurate our model is is by referencing a confusion matrix. This is a table used to describe the performance of the model. The confusion matrix will show us how many of each of the below groupings the model gives us:

* **True Positives (TP)** - # of observations where model predicted a high rating and it actually was a high rating
* **True Negatives (TN)** - # of observations where model predicted a not a high rating and it actually was not a high rating
* **False Positive (FP)** - # of observations where model predicted a high rating but it actually was not a high rating
* **False Negative (FN)** - # of observations where model predicted a not high rating and it actually was a high rating

The below function was used to create the confusion matrix for each model:
```
def confusion_matrix_plot(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''This function will create a confusion matrix chart which shows the accuracy breakdown of the given model.
    Inputs:
        cm: confusion matrix function for tested and predicted y values
        classes: variables to identify for each class (0 or 1)
        normalize: if True will normalize the data, if False will not normalize the data
        title: title of the chart
        cmap: colors to plot
    Returns:
        A graph that represents the confusion matrix
    '''
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add title and labels:
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add axis scales and tick marks:
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add labels to each cell:
    thresh = cm.max() / 2.
    # Iterate through confusion matrix and append labels to the plot:
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        
    # Add legend:
    plt.colorbar()
    plt.show()
    
# Use High Rating and Low Rating  for 0 and 1 classes 
class_names = ['High Rating','Low Rating']
```

The first model I ran was the Decision Tree model. This model uses a tree-like structure to classify by efficiently partitioning each sample into sets with similar data points until you get to a homogenous set and can reasonably predict values for new data. Prior to running each model, I first used a grid search function to determine the optimal hyperparameters for each individual model. This step runs through a variety of combinations of hyperparameters to help ensure we get a more accurate model. For the Decision Tree model, the hyperparameters I included were:
* criterion: measures the quality of the split by looking at impurity either through gini or entropy
* max_depth: maximum number of levels of a tree
* min_samples_split: minimum number of samples required to split an internal node

```
# Determine optimal parameters:
# Declare a baseline classifier:
dtree = DecisionTreeClassifier()

# Create the grid parameter:
grid_dtree = {'criterion': ["gini", "entropy"],
              'max_depth': range(1,10), 
              'min_samples_split': range(1,10)}


# Create the grid:
gs_dtree = GridSearchCV(estimator=dtree, 
                          param_grid=grid_dtree, 
                          scoring='accuracy', 
                          cv=5)

# Fit using grid search:
gs_dtree.fit(X_train, y_train)

# Print best accuracy and best parameters:
print('Best accuracy: %.3f' % gs_dtree.best_score_)
print('\nBest params:\n', gs_dtree.best_params_)
```

This results of this grid search were:
* Best accuracy: 0.675
* Best params:  {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 4}

I then put these best parameters into the decision tree function, fit the model, and then predicted the results as seen below.
```
# Create the classifier, fit it on the training data and make predictions on the test set:
d_tree = DecisionTreeClassifier(criterion='gini',max_depth=9,min_samples_split=4, random_state=123)
d_tree = d_tree.fit(X_train, y_train)
y_pred_test_d_tree = d_tree.predict(X_test)
y_pred_train_d_tree = d_tree.predict(X_train)

print('Decision Tree Train Accuracy: ', accuracy_score(y_train, y_pred_train_d_tree)*100,'%')
print('Decision Tree Test Accuracy: ', accuracy_score(y_test, y_pred_test_d_tree)*100,'%')

# Print classification report:
print(classification_report(y_test, y_pred_test_d_tree))

# Confusion Matrix for Decision Tree
cm_dtree = confusion_matrix(y_test,y_pred_test_d_tree)
confusion_matrix_plot(cm_dtree, classes=class_names, title='Decision Tree Confusion Matrix')
```

The outcome was:
![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/DTREE%20Classification%20Report.png?raw=true)

A similar process was followed for the other four models I looked into. The accuracy of all models came out to be:

### Accuracy:
<br>Decision Tree Test Accuracy:  70.22900763358778 %<br>
<br>Random Forest Test Accuracy:  75.65733672603902 %<br>
<br>Adaboost Test Accuracy:  74.72434266327396 %<br>
<br>XGBoost Test Accuracy:  75.57251908396947 %<br>
<br>Logistic Regession Test Accuracy:  74.6395250212044 %<br>


### Weighted F1 Score:
<br>Decision Tree F1 Score:  0.7719298245614035<br>
<br>Random Forest F1 Score:  0.818008877615726<br>
<br>Adaboost F1 Score:  0.8077419354838709<br>
<br>XGBoost F1 Score:  0.8167938931297711<br>
<br>Logistic Regession F1 Score:  0.8072211476466795<br>

With this, Random Forest was our most accurate model, closely followed by XGBoost. Once the best, most accurate model was chosen, I then took a look at which 30 features were most important within this model. I used the below function to determine the most important features.
```
# Define function to show most important features both visually and in list form
def plot_feature_importances(df, model, num=30, return_list=False):
    '''
    Inputs:
       df : Dataframe to use 
       model : Specific Classification model to extract feature importances from
       num : Number of features that should be plotted
       return_list : If true, returns a list of the features identified as important
    
    Returns:
        Plots a bar graph showing feature importances of num features. Graph is in 
        descending order of feature importances
        If return_list = True, a list of sorted num features will be provided as well.
        
    '''
    feat_imp = dict(zip(df.columns, model.feature_importances_))
    data = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:num]
        
    plt.barh([x[0] for x in data], [x[1] for x in data], align='center', color='darkolivegreen') 
    plt.xlabel("Feature importance", size=14)
    
    if return_list:
        return data
```
 The below code actually ran the above function:
```
# Plot the 30 most influential features based on XGBoost model:
influential_features = pd.Series(rforest.feature_importances_, index=X.columns)
influential_features.nlargest(30).sort_values().plot(kind='barh', color='slategrey', figsize=(10,10))
plt.title('Feature Importances with Random Forest');
```

![](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis/blob/main/Images/Feature_Importance.png?raw=truehttp://)
Within this model, the most influential features include a few neighborhoods such as Midtown West, Morningside Heights, and Greenwich Village along with having more than 31 inspection grade visits or 11-20 visits, receiving or not receiving a critical flag, and the price value of 2 dollar signs.

## Conclusion/Recommendations
**Hypothesis Test Conclusions:**
Based on these results, there are a few recommendations I would give to current or prospective restaurant owners:

* Consider a 4 dollar sign price level rather than a 2 dollar sign price level as these types of restaurants often receive better inspection grades.
* Ensure your restaurant is up to code and has minimal violations so that you are more likely to receive a better inspection grade, which likely will lead to a greater number of Yelp reviews, which in turn can draw more customers into your restaurant.
* Ensure customers have an enjoyable experience at your restaurant so that they will not only give a high Yelp rating, but will also leave a positive review which can encourage other potential customers to try your restaurant.
* When trying to ensure a strong inspection grade, cuisine type and neighborhood do not play a significant factor, so no limitations need to be considered in respect to these two aspects.

**Machine Learning Conclusions:**
From this analysis, a couple of things I would recommend to current or prospective restaurant owners include:

* If possible, consider opening a restaurant in Midtown West or Greenwich Village as restaurants in these neighborhoods tend to receive higher Yelp ratings and higher ratings can lead to drawing in more customers. Additionally, avoid opening a restaurant in Morningside Heights as these tend to receive lower Yelp ratings. 
* Avoid receiving a critical violation flag in an inspection as having one of these violations likely leads to lower Yelp ratings, while not having a critical violation flag likely leads to higher Yelp ratings.

And with that, I have finished the analysis! 

For additional information, here is the link to the full analysis on GitHub: [NYC Yelp and Inspection Data Analysis](https://github.com/rspiro9/NYC-Restaurant-Yelp-and-Inspection-Analysis)

**WATCHOUT NOTE:**
One watchout I want to mention is that the data for this analysis was pulled in August 2020, during the Coronavirus pandemic. This pandemic has hit restaurants especially hard, with many restaurants temporarily or permanently closing down. It seems as though Yelp has done a descent job of identifying resturants that are open vs. closed, though I would guess that the data is not 100% accurate as restaurants' statuses were constantly changing during this time. Therefore, the dataset may include some restaurants that are no longer in business, or may have falsely excluded some restaurants that were in business.
