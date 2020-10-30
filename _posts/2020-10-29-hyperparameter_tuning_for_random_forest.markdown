---
layout: post
title:      "Hyperparameter Tuning For Random Forest"
date:       2020-10-30 00:41:55 +0000
permalink:  hyperparameter_tuning_for_random_forest
---


In this blog post I will discuss how to do hyperparamter tuning for a classification model, specifically for the Random Forest model.   So what exactly is hyperparameter tuning? In Machine Learning, a hyperparameter is a paramater that can be set prior to the beginning of the learning process. Different classification methods have different hyperparamters. We are going to be focusing on Random Forest Classification, which is an ensemble method for decision trees that both trains trees on different samples of data (bagging) and randomly selects a subset of features to use as predictors (subspace sampling method) to create a 'forest' of decision trees. This method typically gives better predictions than any single decision tree would give. This image provides a visual representation of how Random Forests work. ![](https://miro.medium.com/max/1400/1*58f1CZ8M4il0OZYg2oRN4w.png)

(image source: https://medium.com/@ar.ingenious/applying-random-forest-classification-machine-learning-algorithm-from-scratch-with-real-24ff198a1c57)

There are a variety of hyperparameters that can be tuned for Random Forests. A full list can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on the scikit-learn page for Random Forests. The six that we are going to review in more depth are:

1. n_estimators
2. criterion
3. max_depth
4. max_features
5. min_samples_split
6. min_samples_leaf

## 1. n_estimators:
The n_estimators hyperparameter specifices the number of trees in the forest. For example, if n_estimators is set to 5, then you will have 5 trees in your Forest. The default value was updated to be 100 while it used to be 10. Having more trees can be beneficial as it can help improve accuracy due to the fact that the predictions are made from a larger number of "votes" from a diverse group of trees, but the more trees you have, the easier it is to run into large computational expenses.

## 2. criterion:
This measures the quality of the split by looking either  at '*gini*' or '*entropy*'.  Gini looks at gini impurity, which measures the frequency that a randomly chosen element would be labelled incorrectly. Entropy looks at information gain, which gauges the disorder of a grouping. The default value is *gini*.


## 3. max_depth:
Max_depth identifies the maximum number of levels of a tree. In other words, what it the longest path between the root node and the leaf node. The default value is '*none*', which means the tree keeps expanding until all leaf nodes are pure (all data on the leaf is from the same class).  Having a larger number of splits allows for each tree to better explain the variation in the data, but too many splits could lead to overfitting the data.

## 4. max_features:
This represents the maximum number of features to consider for splitting a node. With a larger number of features to choose from to determine the best split, model performance can improve, but this can also make the trees less diverse, therefore causing overfitting. A few common options for this hyperparameter are:
* *int* - any specified integer value
* *auto* - no restrictions given (this is the default)
* *sqrt* - square root of the total number of features
* *log2* - base 2 logarathim of the total number of features

## 5. min_samples_split:
The min_samples_split hyperparameter is the minimum number of samples required to split an internal node. The default is 2, which would mean that at least 2 samples are needed to split each internal leaf node. By increasing the number of this hyperparameter, we are reducing the number of splits happening in the decision tree, which helps to prevent overfitting. However, having too large of a value can lead to underfitting as the tree may not be able to split enough times to get to pure nodes.

## 6. min_samples_leaf:
Min_samples_leaf represents the minimum number of samples required for a leaf node.  For example, if this parameter is set to 5, then each leaf must have at least 5 samples that it classifies. The default value is 1. The helps to prevent the growth of the tree, which can prevent overfitting. But similar to min_samples_split, having too large of a value can lead to underfitting due to the tree not being able to split enough times to get to pure nodes.

## How do you choose which hyperparameters to use?
The optimal set of hyperparameters will vary on a case by case basis for each analysis being run. Therefore, to determine the best hyperparameters, multiple options should be considered. However, trying to change each hyperparameter one at a time and then comparing each outcome would be extremely time consuming if done manually. Luckily, there are methods in place to make this process more efficient, such as using an exhaustive grid search.  In an exhaustive grid search, you provide a list of all of the different hyperparameters you want to consider and the grid search will then try every single possible combination of these metrics. From scikit-learn you can use the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) method, which will do an exhaustive grid search and optimize the parameters by cross-validating a grid search over the parameter grid. Cross validation will help ensure that we are getting the model that performs the best it can given the hyperparameters it was trained on. Within this function, you can specify the number of folds you want to use in your cross-validation splitting strategy using the *cv* parameter. It is also common to use the *n_jobs* parameter, which identifies the number of jobs to use in parallel. This can help shorten the amount of time it takes to run this function. I am using *njobs = -1*, which means all processors will be used.

To run this grid search process, we first declare the RandomForestClassifier(). Next, list out the parameters we want to optimize. Then, create the grid utilizing the GridSearchCV function. We then fit the model and finally call the best parameters and their corresponding best accuracy.


```
# Determine optimal parameters:
# Declare a baseline classifier:
forest = RandomForestClassifier()

# Create the grid parameter:
grid_rf = {'n_estimators': [100, 120, 150],
           'criterion': ['entropy', 'gini'], 
           'max_depth': [None,1,3,5,7,9],
           'max_features': range(1,11),  
           'min_samples_split': range(2, 10),
					 'min_samples_leaf': [1,3,5]}

# Create the grid:
gs_rf = GridSearchCV(forest, grid_rf, cv=3, n_jobs=-1)

# Fit using grid search:
gs_rf.fit(X_train, y_train)

# Print best accuracy and best parameters:
print('Best accuracy: %.3f' % gs_rf.best_score_)
print('\nBest params:\n', gs_rf.best_params_)
```

For this data, which represents a dataset of yelp restaurant reviews in Manhattan along with information about corresponding restaurant inspection grades, our best parameters are:

* n_estimators: 120
* criterion: Entropy
* max_depth: None
* max_features: 10
* min_samples_split: 8
* min_samples_leaf: 3

And our best accuracy: is 0.746.


Once we have identified our best parameters, then we can put this parameters into our Random Forest Classifier, fit the model, and determine the overall accuracy of our optimized Random Forest model, as seen below.

```
# Create the classifier, fit it on the training data and make predictions on the test set:
rforest = RandomForestClassifier(n_estimators=120, criterion = 'entropy', max_depth=None, max_features=10, 
                                 min_samples_split=8, 'min_samples_leaf'=3, random_state=123)
rforest.fit(X_train, y_train)
y_pred_test_rf = rforest.predict(X_test)
y_pred_train_rf = rforest.predict(X_train)

# Check the Accuracy:
print('Random Forest Train Accuracy: ', accuracy_score(y_train, y_pred_train_rf)*100,'%')
print('Random Forest Test Accuracy: ', accuracy_score(y_test, y_pred_test_rf)*100,'%')
```

Random Forest Train Accuracy:  92.05274411974341 %

Random Forest Test Accuracy:  71.82044887780549 %

### So, is adjusting hyperparameters necessary?
In addition to running the Random Forest model with adjusting the hyperparameters, I also ran the model without any adjustments. With this run, I got an test accuracy of 70.90606816292602 %, which is slightly lower than the test accuracy from our tuned model. Therefore, in this case, it does seem like adjusting the hyperparameters improved our model.

## Conclusion:
In all, hyperparameter tuning is a great way to have a better chance at identifying the best possible combination of parameters and therefore having the most accurate model possible. Doing an exhaustive grid search is a great way to try as many different combinations of hyperparameters as posisble in order to achieve this. However, having a large number of hyperparameters could be very time consuming and costly as the grid search could have to run through thousands or millions of different possible combinations. Therefore, it may be more efficient to start with a few hyperparameters to see how they shift the results, and then determine if it is worth expanding your parameters further.




