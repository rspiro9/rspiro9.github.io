---
layout: post
title:      "Decision Tree Classification Guide"
date:       2020-04-22 20:55:36 -0400
permalink:  decision_tree_classification_guide
---


Decision trees are a Supervised Machine Learning algorithm used to classify data. Decision Trees are very popular because they are effective and easy to visualize and interpret.  Essentially, a decision tree is a flowchart where each level represents a different yes/no question. There are 3 different parts to a tree:
1. Internal Nodes - each represents a 'test' on an attribute (i.e. does a coin flip end up heads or tail)
2. Edges/Branches - the outcome of the test
3. Leaf Nodes - predicts outcome by classifying data (this is a terminal node meaning there are no further nodes/branches)

As an example for how to use a decision tree, let's say you want to play a game of soccer. In order to play outside, you need nice weather. A decision tree can be used to predict whether or not you can play soccer based on different weather features as shown below. 

![](https://cdn.educba.com/academy/wp-content/uploads/2019/05/weather-Decision-Tree-Algorithm.png)

*Source*: https://www.educba.com/decision-tree-algorithm/

Here the internal nodes are questions such as 'is it sunny, rainy, or cloudy?' or 'is there high humidity?'. The edges/branches are the answers bringing you to the next node (i.e. it is sunny or it is cloudy). And the leaf nodes are the yes and no at the bottom of the diagram telling you if you can play outside or not. So, the algorithm starts with the top question of 'is it sunny, rainy, or cloudy?' and makes a decision (a.k.a. the answer to the question). Then based off this decision, moves on to the corresponding next question. This continues until we have a final result and our data has been classified appropraitely. 

**Training Decision Tree:**

It is important to train your decision tree in order to make sure that you are selecting the correct features to make a split on as this could effect the efficacy of the model. Training uses 'recursive binary splitting,' which considers all features and different split points are tried and tested based on a designated cost function. It selects the split with the lowest cost. The process to train data consists of:
1. Use training dataset with features/predictors and a target.
2. Make splits for the target using the values of predictors. Feature selection is used to determine which features to use as predictors.
3. Grow tree until a stopping criteria is met.
4. Show the tree a new set of features using an unkown class. The resulting leaf node is the class prediction for this example.

**Creating a Decision Tree:**

Let's walk through an example of how to run a decision tree classifier using scikit-learn. For this example, I will be using a dataset containing information on Terry Stops. A terry stop is when a police officer stops a person based on 'reasonable suspicion' that the person may have been involved in criminal activity.  We will be using this data to predict whether or not a terry stop ends in an arrest. Once all of the necessary pre-processing of the data has been done, we can begin by finding the best parameters to minimize our cost function. Trying different parameters is a great way to tune our algorithm. For decision trees, here are a few parameters to consider adjusting:
* DecisionTreeClassifier - measures the quality of the split using impurity measures such as 'Gini' or 'Information Gain'.
* max_depth - how deep to make the tree. If 'None', nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
* min_samples_split - sets the minimum number of samples needed to split a node.
* min_samples_leaf - identifies minimum number of samples we want a leaf node to contain. Once achieved at a node, there are no more splits.
* max_features - number of features to consider when looking for the best split.

These are just a few of the many parameters you can set. Rather than having to try different combinations of these parameters individually, we can create a grid search function to run through all of the different combinations based off the parameter inputs we provide. To do so, we have to provide the classifier we want to use and the parameter inputs, as shown below. 

```
# Determine optimal parameters:
# Declare a baseline classifier:
dtree = DecisionTreeClassifier()

# Create a parameter grid and grid search to identify the best parameters:
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": range(1,10),
    "min_samples_split": range(2,10)
}
gs_tree = GridSearchCV(dtree, param_grid, cv=5, n_jobs=-1)

# Fit the tuned parameters:
gs_tree.fit(X_train_ohe, y_train)

# Print best estimator parameters:
print(gs_tree.best_params_)
```

Running this function told us that the best parameters based on the inputs we provided are a gini criterion with max_depth = 1 and min_samples_split = 2. Once we have this info, we can go ahead with creating our decision tree classifier using these parameters. We then fit the model to the training data and predict the testing data.
```
# Create the classifier, fit it on the training data and make predictions on the test set:
d_tree = DecisionTreeClassifier(criterion='gini',max_depth=1,min_samples_split=2)
d_tree = d_tree.fit(X_train_ohe, y_train)
y_pred = d_tree.predict(X_test_ohe)
```

Now that we have our algorithm run, we need to pull a few metric to determine how well our model has performed. There are 2 different variables I am going to look at. The first is the accuracy score. This tells you how accurate your classification model is.
```
print('Decision Tree Accuracy: ', accuracy_score(y_test, y_pred)*100,'%')
```

We get an accuracy score of 79.95872884853488 %.  Meaning, our model can correctly predict when an arrest was made from a terry stop 79.96% of the time. I then look at a confusion matrix. A confusion matrix is a table used to describe the performance of a model, and it is a good way to visualize how accurate our model is. The confusion matrix will show us how many of each of the below groupings the model gives us:
* True Positives (TP) - # of observations where model predicted person was arrested and they actually were arrested
* True Negatives (TN) - # of observations where model predicted person was not arrested and they actually were not arrsted
* False Positive (FP) - # of observations where model predicted person was arrested but they are actually were not arrested
* False Negative (FN) - # of observations where model predicted person was not arrested and the actually were arrested

Below is a function that can be used to create a confusion matrix:
```
# Define Confusion Matrix:
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
```

Now to run this matrix:
```
# Use Arrested and Not Arrested for 0 and 1 classes 
class_names = ['Arrested','Not Arrested']
# Confusion Matrix for Decision Tree
cm_dtree = confusion_matrix(y_test,y_pred)
confusion_matrix_plot(cm_dtree, classes=class_names, title='Decision Tree Confusion Matrix')
```

![](data:image/png;base64,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)

We see here that we have over 9,000 true positives (top left corner) and over 400 true negatives (bottom right corner). However we do have over 2000 incorrect classifications (bottom left and top right corners), indicating there is still room to improve this model.

