# Financial Indicators of US Stocks – Multiple Machine Learning Models Analysis

This Jupyter Notebook provides an analysis to see if financial indicators that are present in 10-K filings of public companies can be used to predict the stock performance at the end of the year.  We compare the results using different machine learning models and techniques.

---

## Data Sources

[200+ Financial Indicators of US stocks](https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018)

---

## Technologies

This analysis is a Jupyter Notebook that makes use of the following Python libraries:
* Pandas
* Numpy
* Autogluon tablular: `TabularPredictor`, `TabularDataset`
* imblearn `RandomOverSampler`
* sklearn model_selection functions such as `train_test_split`
* sklearn preprocessing: `OneHotEncoder`, `StandardScaler`, `MinMaxScaler`
* sklearn `SimpleImputer`
* sklearn metrics: `confusion_matrix`, `classification_report`
* sklearn ensemble models: `RandomForestClassifier`, `AdaBoostRegressor`
* sklearn svm
* sklearn linear model: `LogisticRegression`
* Tensorflow 2.0
* Keras


---

## Installation Guide

To use this notebook:
* Install Jupyter Lab Version 2.3.1 and Python 3.7.
* Install Autogluon
* Install imbalanced-learn
* Install scikit-learn
* Install Tensorflow (Keras is included in Tensorflow 2.0)

Open the notebook in Jupyter Lab and you can rerun the analysis.

---

## Data
### Features
Our dataset contains feature data in the form of financial indicators that are present in 10-K filings of public companies.  We separated our feature dataset into "raw" and "key" datasets, where the raw data contained all the fields from the original dataset, and the "key" dataset contained 6 key financial values/ratios.

### Targets
Our dataset contains a `2019 PRICE VAR [%]` that gives the percent price variation for the year (meaning from the first trading day to the last trading day for that year).  This column is used for regression modeling.

Our dataset also contains a `Class` column which is used for classification.  For each stock, if the `PRICE VAR [%]` value is positive, class is 1 and the stock went up for the year.  If the `PRICE VAR [%]` value is negative, class is 0, and the stock went down for the year.

### Precision vs. Recall
Because we are predicting stock prices, precision will be more important than recall because the cost of acting is high, but the cost of not acting is low.

Furthermore, we expect to act only on Class 1, which represents stocks that go up.  So we optimized our classification modeling for Precision of Class 1, meaning it is most important that when we predict that a stock is one that should go up, we are correct.

### Data cleaning strategies
* We tried various strategies for how much missing/invalid data we discarded.
* We tried replacing zeros with the mean for each column, and it had subtle/questionable effects.  Later, when we tried it again after some of our other changes, it had almost no effect on most of our models, except for a noticeably bad effect on our Tensorflow models that used the raw dataset.
* We tried using the experimental scikit-learn `IterativeImputer` to impute NaN values.  We tried it with various estimators:
  * BayesianRidge: regularized linear regression
  * DecisionTreeRegressor: non-linear regression
  * KNeighborsRegressor: comparable to other KNN imputation approaches
  * The first two models had trouble converging, and the K-neareset neigbors estimator produced results similar to just using the column mean.  So we stuck with the `SimpleImputer` using the column mean.
* We tried dropping some of the redundant columns but in general when we saw our Class 1 accuracy go down for multiple models, so we left in all the columns from the raw dataset.
* We also tried scaling data to be between 0 and 1 using the scikit-learn `MinMaxScaler`.  This caused some of our models to only predict Class 1, but it had a positive effect on the Tensorflow models, especially with the raw dataset, so we used the 0 to 1 scaled data for those models.

### Final data cleaning choices
* Removing rows and columns with over 25% NaN values.
* Removing rows with over 25% zeros and columns with over 50% zeros.
* Encoding categorical data using one-hot encoding.
* Replacing remaining NaN values by imputing them with the mean for each column.
* Scaling the feature data by removing the mean and scaling to unit variance.
* Scaling feature data again for another dataset, scaling it to be in the range of 0 to 1.  Using this dataset for Tensorflow models with the raw dataset.

## Modeling
### Autogluon
Autogluon was used as a baseline model.  Because Autogluon has its own strategies for dealing with NaN data, we used the financial dataframe where rows and columns with too many NaNs and zeros were removed, but no replacement of NaN values or encoding of categorical data.  In general AutoGluon handled the cleaning, encoding, creating and fitting of multiple models.

#### Results
```
              precision    recall  f1-score   support

           0       0.60      0.36      0.45       280
           1       0.77      0.90      0.83       674

    accuracy                           0.74       954
   macro avg       0.69      0.63      0.64       954
weighted avg       0.72      0.74      0.72       954
```

### Classification SVM SVC model using raw dataset
#### Results
```

              precision    recall  f1-score   support

           0       0.54      0.14      0.22       280
           1       0.73      0.95      0.82       674

    accuracy                           0.71       954
   macro avg       0.63      0.55      0.52       954
weighted avg       0.67      0.71      0.65       954 
```
### Classification SVM SVC model using raw dataset with oversampling of minority class
For oversampling we used the imblearn `RandomOverSampler`.

#### Results
```

              precision    recall  f1-score   support

           0       0.46      0.72      0.56       280
           1       0.85      0.64      0.73       674

    accuracy                           0.67       954
   macro avg       0.65      0.68      0.64       954
weighted avg       0.73      0.67      0.68       954
```

### Classification SVM SVC model using key dataset
#### Results
```

              precision    recall  f1-score   support

           0       0.67      0.01      0.01       280
           1       0.71      1.00      0.83       674

    accuracy                           0.71       954
   macro avg       0.69      0.50      0.42       954
weighted avg       0.70      0.71      0.59       954
```

### Classification Random Forest Classifer model using raw dataset
#### Results
```

              precision    recall  f1-score   support

           0       0.62      0.38      0.47       280
           1       0.78      0.91      0.84       674

    accuracy                           0.75       954
   macro avg       0.70      0.64      0.65       954
weighted avg       0.73      0.75      0.73       954
```

### Classification Random Forest Classifer model using raw dataset with oversampling of minority class
#### Results
```

              precision    recall  f1-score   support

           0       0.58      0.47      0.52       280
           1       0.80      0.86      0.83       674

    accuracy                           0.75       954
   macro avg       0.69      0.66      0.67       954
weighted avg       0.73      0.75      0.74       954
```

### Classification Random Forest Classifer model using key dataset
#### Results
```

              precision    recall  f1-score   support

           0       0.67      0.01      0.01       280
           1       0.71      1.00      0.83       674

    accuracy                           0.71       954
   macro avg       0.69      0.50      0.42       954
weighted avg       0.70      0.71      0.59       954
```

### Regression AdaBoost model using raw dataset
#### Results (1.0 is best)
Coefficient of determination of the prediction: -0.57193532905021

### Regression AdaBoost model using key dataset
#### Results (1.0 is best)
Coefficient of determination of the prediction: -1.6260137253640794

### Classification Logistic Regression model using raw dataset
#### Results
```

              precision    recall  f1-score   support

           0       0.56      0.25      0.35       280
           1       0.75      0.92      0.82       674

    accuracy                           0.72       954
   macro avg       0.66      0.58      0.59       954
weighted avg       0.69      0.72      0.68       954
```

### Classification Shallow Neural Network using key dataset with oversampling of minority class
#### Results
```
              precision    recall  f1-score   support

           0       0.48      0.55      0.51       280
           1       0.80      0.76      0.78       674

    accuracy                           0.69       954
   macro avg       0.64      0.65      0.65       954
weighted avg       0.71      0.69      0.70       954

```

### Classification Deep Neural Network using key dataset with oversampling of minority class
#### Results
```

              precision    recall  f1-score   support

           0       0.45      0.61      0.51       280
           1       0.81      0.69      0.74       674

    accuracy                           0.66       954
   macro avg       0.63      0.65      0.63       954
weighted avg       0.70      0.66      0.68       954
```

### Classification Shallow Neural Network using raw dataset scaled between 0 and 1, with oversampling of minority class
#### Results
```

              precision    recall  f1-score   support

           0       0.47      0.75      0.57       280
           1       0.86      0.65      0.74       674

    accuracy                           0.68       954
   macro avg       0.66      0.70      0.66       954
weighted avg       0.74      0.68      0.69       954
```

### Classification Deep Neural Network using raw dataset scaled between 0 and 1, with oversampling of minority class
#### Results
```

              precision    recall  f1-score   support

           0       0.39      0.86      0.54       280
           1       0.89      0.44      0.59       674

    accuracy                           0.57       954
   macro avg       0.64      0.65      0.56       954
weighted avg       0.74      0.57      0.58       954
```
## Conclusion
Although several of our models peformed fairly well above 50% Precision for Class 1, the model that did the best was our Deep Neural Network classification model that used the raw dataset scaled between 0 and 1, and used oversampling of the minority class.

This was our most complex model and took longer to run than most of the others.  This model gave a Precision for Class 1 of 0.89, meaning if we use this model to predict that a stock is expected to go up in a year, there is an 89% chance that our classification was correct.

Our second best model was an SVM SVC classification model that used the raw dataset scaled by removing the mean and scaling to unit variance, again with oversampling of minority class.  This model ran more quickly then the Tensorflow model and gave a Precision for Class 1 of 0.85.

Autogluon provided the most automated solution, as we could essentially put the raw data into it and it handled the cleaning, encoding, creating and fitting of multiple models.  The Autogluon model gave a Precision for Class 1 of 0.77.

We were not able to get any highly accurate regression models, and next steps for our project are to look into regression modeling more as it might provide interesting insights into the performance of individual stocks that went up or down, since binary classification marks stocks as either "stocks to buy" or "stocks not to buy" without any nuance.

---

## Contributors

* Allen Wong

* Christine Guo 

* LaNaya Johnson 

* Michael Danenberg

---

## License

MIT