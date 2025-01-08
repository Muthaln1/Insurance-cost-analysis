### Introduction:
+ The dataset used in this analysis contains information on individual medical insurance bills. Since the charges are continuous and positive, they are well-suited for modeling with linear regression. Predicting medical costs is crucial, as it enables hospitals to forecast revenue and plan for the necessary procedures required by their patient population

### Goal:
+ To analyze how these different characteristics like `age`, `bmi`, `sex`, `region` etc relate to the total medical cost
+ To construct the linear regression predictive model for the insurance cost, given the information about the patient

### Dataset contains:
+ `age`     : The age of the individual
+ `sex`     : The gender of the individual.It is typically encoded as a binary categorical variable(female, male)
+ `bmi`     : The body mass index (BMI) of the individual. This is a continuous variable, calculated as weight (in kg) divided by height
+ `children`: The number of children or dependents the individual has. This is a discrete, non-negative integer value(
+ `smoker`  : Indicates whether the individual is a smoker. This is a categorical variable, where 'yes' means the individual smokes and 'no'               means they do not
+ `region`  : The region where the individual resides. This is typically encoded as a categorical variable, possibly with four regionsin the               US:
     + southeast,
     + southwest,
     + northeast
     + northwest 
+ `charges` : The medical insurance charges, which are continuous, representing the cost of the insurance for the individual. These are                    typically in USD.


```python
# Importing libraries needed for the analysis 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# Read the CSV file into a pandas DataFrame
insurance = pd.read_csv('insurance.csv')
print(insurance.head(2))
```

       age     sex    bmi  children smoker     region     charges
    0   19  female  27.90         0    yes  southwest  16884.9240
    1   18    male  33.77         1     no  southeast   1725.5523



```python
# Detect missing values
print(insurance.isnull().sum())
```

    age         0
    sex         0
    bmi         0
    children    0
    smoker      0
    region      0
    charges     0
    dtype: int64



```python
# Use info() to get a concise summary of the DataFrame
print(insurance.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB
    None



```python
# Using describe() to get summary statistics of numerical columns
print(insurance.describe())
```

                   age          bmi     children       charges
    count  1338.000000  1338.000000  1338.000000   1338.000000
    mean     39.207025    30.663397     1.094918  13270.422265
    std      14.049960     6.098187     1.205493  12110.011237
    min      18.000000    15.960000     0.000000   1121.873900
    25%      27.000000    26.296250     0.000000   4740.287150
    50%      39.000000    30.400000     1.000000   9382.033000
    75%      51.000000    34.693750     2.000000  16639.912515
    max      64.000000    53.130000     5.000000  63770.428010


### Data Analysis:
+ The insurance data is evenly split between male and female customers.
+ Non-smokers make up 79% of the total customer base, while smokers account for 20%.
+ The data is evenly distributed across the four regions:
  + Southeast
  + Southwest
  + Northwest
  + Northeast


```python
# Use value_counts() to count unique values in the 'sex' column
sex_count = pd.DataFrame(insurance['sex'].value_counts())
smoker_count = pd.DataFrame(insurance['smoker'].value_counts())
region_count = pd.DataFrame(insurance['region'].value_counts())
```


```python
# Plotting charts for categorical columns
fig,(ax) = plt.subplots(3,1,figsize=(8,10))
sex_count.plot(kind='bar',ax=ax[0],color = 'grey')  # int data type
ax[0].tick_params(axis='x', labelrotation=0)
ax[0].set_xlabel('sex')
ax[0].set_ylabel('sex_count')
ax[0].set_title('sex_count')
for index, value in enumerate(sex_count['count']):
    ax[0].text(index, value + 10, str(value), ha='center')
    
smoker_count.plot(kind='bar',ax=ax[1],color ='orange') # Object data type
ax[1].set_xlabel('smoker')
ax[1].set_ylabel('smoker_count')
ax[1].tick_params(axis='x', labelrotation=0)
ax[1].set_title('smoker_count')
for ind, val in enumerate(smoker_count['count']):
    ax[1].text(ind, val + 10, str(val), ha='center')

region_count.plot(kind='bar',ax=ax[2])  # Object data type
ax[2].set_xlabel('region')
ax[2].set_ylabel('region_count')
ax[2].tick_params(axis='x', labelrotation=0)
ax[2].set_title('region_count')
for ind, val in enumerate(region_count['count']):
    ax[2].text(ind, val + 10, str(val), ha='center')

plt.tight_layout()
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Output%201.png)
    



```python
# Data equally divided between the genders
print(insurance['sex'].value_counts())
```

    sex
    male      676
    female    662
    Name: count, dtype: int64



```python
# Number of non smokers are more than the smokers
print('(in percentage):\n',insurance['smoker'].value_counts(normalize = True)*100)
```

    (in percentage):
     smoker
    no     79.521674
    yes    20.478326
    Name: proportion, dtype: float64



```python
# Data equally divided between regions
print(insurance['region'].value_counts())
```

    region
    southeast    364
    southwest    325
    northwest    325
    northeast    324
    Name: count, dtype: int64


### Mapping the categorical values with Binary values for analysis:
+ Mapping the categorical values with binary values for better analysis as the non smoker is considered healthy compared to a smoker
+ Non smoker provided with the rank of 0 and smoker with 1
+ Male provided with binary value 1 and female with binary value 0


```python
# Apply map function to each element in the 'smoker' column
category_map = {'yes': 1, 'no': 0}
insurance['smoker'] = insurance['smoker'].map(category_map)

# Apply map function to each element in the 'sex' column
category_map1 = {'female': 0, 'male': 1}
insurance['sex'] = insurance['sex'].map(category_map1)
print(insurance.head(2))
```

       age  sex    bmi  children  smoker     region     charges
    0   19    0  27.90         0       1  southwest  16884.9240
    1   18    1  33.77         1       0  southeast   1725.5523


### Correlation Analysis : Numerical data:
+ The outcome Hospital `Charges` highly correlates with the features `age`(0.299008), `bmi`(0.198341) and `smoker`(0.787251).


```python
# Calculate Pearson correlation matrix
correlation_data = insurance[insurance.select_dtypes(include=['int64','float64']).columns.values].corr()
print(correlation_data)
```

                   age       sex       bmi  children    smoker   charges
    age       1.000000 -0.020856  0.109272  0.042469 -0.025019  0.299008
    sex      -0.020856  1.000000  0.046371  0.017163  0.076185  0.057292
    bmi       0.109272  0.046371  1.000000  0.012759  0.003750  0.198341
    children  0.042469  0.017163  0.012759  1.000000  0.007673  0.067998
    smoker   -0.025019  0.076185  0.003750  0.007673  1.000000  0.787251
    charges   0.299008  0.057292  0.198341  0.067998  0.787251  1.000000



```python
# Create the heatmap and importing respective libraries
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(correlation_data, cmap="Blues",cbar=True,annot=True)
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/output%202.png)
    


### Outcome - Hospital `Charges`
+ Hospital charges exhibit outliers when grouped by age category
+ The highest charge of 63,770 is associated with age 54, while the second highest charge of 62,592 belongs to age 45
+ For certain age groups, there is a noticeable gap between the mean and median values. This suggests that incidents, terminal illnesses, or   accidents may be driving the spikes in the mean, resulting in outliers
+ Charges related to general health care are significantly different from those tied to unexpected accidents or incidents


```python
# `Charges` grouped by `age` feature
age_charges = insurance.groupby('age').agg(mean = ('charges','mean'), median =('charges','median'),max=('charges','max'),min=('charges','min')).reset_index()
print(age_charges.sort_values(ascending = False,by = 'max').head(10))
```

        age          mean        median          max          min
    36   54  18758.546475  11816.449500  63770.42801   9850.43200
    27   45  14830.199856   8603.823400  62592.87309   7222.78625
    34   52  18256.269719  11396.900200  60021.39897   9140.95100
    13   31  10196.980573   4738.268200  58571.07448   3260.19900
    15   33  12351.532987   6210.083300  55135.40209   3704.35450
    42   60  21979.418507  13204.285650  52590.82939  12142.57860
    10   28   9069.187564   4344.951450  51194.55914   2689.49540
    46   64  23275.530837  15528.758375  49577.66240  13822.80300
    41   59  18895.869532  12928.791100  48970.24760  11743.29900
    26   44  15859.396587   8023.135450  48885.13561   6948.70080


### Distribution of charges
+ Due to the presence of outliers the charges are skewed to the right


```python
# Plot histogram for the 'charges' column
insurance['charges'].hist()
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Histogram of `Charges`')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Histogram%20of%20charges.png)
    


### Logarithmic conversion of `charges`
+ Post using log conversion, the distribution of the outcome is symmetrical


```python
# Log conversion and assign them to their columns
insurance['charges'] = np.log2(insurance['charges'])
```


```python
# Calculate skewness
skewness_value = insurance['charges'].skew()
print(f"Skewness: {skewness_value}")
```

    Skewness: -0.09009752473024946



```python
# Plot histogram for the 'charges' column
insurance['charges'].hist()
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Histogram of Charges Data')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Histogram%20of%20charges%20log.png)
    


### Distribution of log `Charges` by `sex`
+ Charges has similar mean value between both male and female


```python
# `Charges` grouped by `sex` feature
sex_charges = insurance.groupby('sex').agg(mean = ('charges','mean'),max=('charges','max'),min=('charges','min')).reset_index()
print(sex_charges.sort_values(ascending = False,by = 'mean'))
```

       sex       mean        max        min
    1    1  13.133981  15.933711  10.131695
    0    0  13.119043  15.960600  10.650612



```python
# Box plot of sex versus Charges
sns.boxplot(x='sex', y='charges', data=insurance)
plt.xticks(rotation=90)
plt.title('Boxplot of Charges by sex')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Box%20plot%20charges%20by%20sex.png)
    


### Distribution of log `Charges` by `Age`
+ Age is directly proportional to the hospital charges
+ Age goes up, charges goes up


```python
# Box plot of `age` versus `charges`
sns.boxplot(x='age', y='charges', data=insurance)
plt.xticks(rotation=90)
plt.title('Boxplot of Charges by Age')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Boxplot%20charges%20by%20age.png)
    


### Distribution of log `charges` by `bmi`


```python
# Box plot of `bmi` versus `charges`
sns.scatterplot(x='bmi', y='charges', data=insurance)
plt.title('scatter plot of bmi by charges')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/scatter%20plot%20of%20bmi%20by%20charges.png)
    


### Distribution of log `charges` by `smoker`
+ The patients who smoke has a higher outcome mean compared to the non smoker
+ The patients who smoke has a floor value (13.64) compared to the non smoker(10.13)


```python
# `charges` grouped by `smoker` feature
smoke_charges = insurance.groupby('smoker').agg(mean = ('charges','mean'),max=('charges','max'),min=('charges','min')).reset_index()
print(smoke_charges.sort_values(ascending = False,by = 'mean'))
```

       smoker       mean        max        min
    1       1  14.865688  15.960600  13.647172
    0       0  12.678739  15.171748  10.131695



```python
# Box plot smoker versus Charges
sns.boxplot(x='smoker', y='charges', data=insurance)
plt.title('Boxplot of Charges by smoking/non smoking data')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Boxplot%20of%20charges%20by%20smoker.png)
    


### Distribution of log `charges` by `region`
+ All the regions has a similar mean outcome


```python
# `Charges` grouped by `region` feature
region_charges = insurance.groupby('region').agg(mean = ('charges','mean'),max=('charges','max'),min=('charges','min')).reset_index()
print(region_charges.sort_values(ascending = False,by = 'mean'))
```

          region       mean        max        min
    0  northeast  13.227737  15.837901  10.726896
    2  southeast  13.160843  15.960600  10.131695
    1  northwest  13.084859  15.873189  10.662971
    3  southwest  13.029121  15.682524  10.277944



```python
# Box plot region versus Charges
sns.boxplot(x='region', y='charges', data=insurance)
plt.title('Boxplot of Charges by Region')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Boxplot%20of%20charges%20by%20region.png)
    


### Segmenting data for test split analysis
+ Features used in the test split function to predict the outcome `charges`:
    + `age`
    + `bmi`
    + `Children`
    + `sex`
+ Residual mean value (-1.9324) closer to zero (difference between actual and predicted values) indicates that the model may fits the data
+ R2 score - 0.297 (or 29.7%) of the variance in the target variable (y) can be explained by the model using the independent variable(s) (e.g., `age` & `bmi`)
+ The relation between the residual and the predictions form a inverted U showing a quadratic relation between them not the linear relationship between them


```python
# Assigning the features and outcome to seperate variables
X = insurance.drop(columns ='charges')
# Log value
y = insurance['charges']
```


```python
# Using test split function to split the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state =472)
```

### Numerical and Binary features


```python
X_train_subset = X_train[['age','bmi','children','sex']]

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_subset ,y_train)

# Intercept
intercept = model.intercept_
print('Intercept:',intercept)

#Co-efficients
coefficient = model.coef_
print('Co-efficients are:',coefficient)
```

    Intercept: 10.454359785657271
    Co-efficients are: [0.04919824 0.01759696 0.15784123 0.0329255 ]



```python
# Use the model to make predictions on the test data
predictions = model.predict(X_train_subset)
```


```python
residuals =  y_train - predictions  # Actual and predicted difference of values

residual_mean = residuals.mean()
print('Residual mean:',residual_mean)
```

    Residual mean: -1.4078042988891704e-15



```python
from sklearn.metrics import r2_score
R2 = r2_score(y_train, predictions)
print('R2 Score:',R2)
```

    R2 Score: 0.31689974625386175



```python
training_mse = mean_squared_error(y_train, predictions)
print('Mean Squared Error:',training_mse)

training_rmse = mean_squared_error(y_train, predictions,squared=False)
print('Root Mean Squared Error:',training_rmse)
```

    MSE: 1.2104539710935946
    Root MSE: 1.1002063311459331


    /Users/nandhinimuthalraj/Documents/anaconda/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(



```python
## Looking for a even band of dots around zero
# Create a scatter plot
plt.scatter(predictions,residuals)
plt.title('Residual vs Predictions')
plt.xlabel('predictions')
plt.ylabel('residuals')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Numerical%20and%20binary%20features.png)
    


### Categorical features
+ Using One hot encoding to include the categorical feature into the analysis
+ Scaling the numerical features as the features are in different scales
+ The features with positive coefficients are age, bmi, children, smoker, region_northeast, and region_northwest, which means that these variables have a positive relationship with the outcome. On the other hand, the features with negative coefficients are sex, region_southeast, and region_southwest, indicating an inverse relationship with the outcome
+ Residual mean value (-1.726) is much closer to zero (difference between actual and predicted values) indicates that the model fits the data
+ R2 score - 0.7742 (or 77.42%) of the variance in the target variable (y) can be explained by the model using the independent variable(s) (e.g., `age` & `bmi`)
+ The relationship between the residuals and the predictions forms a slight inverted U-shape, indicating a quadratic rather than a linear relationship between them


```python
# Creating dummy variable for the categorical feature region
X_train_with = pd.get_dummies(X_train[['region']])

# Scaling the numerical features using standard scaler
from sklearn.preprocessing import StandardScaler

def scaler(df):
    scal = StandardScaler()
    scal.fit(df)
    scal_x = scal.transform(df)
    return scal_x

scaled_data = pd.DataFrame(scaler(X_train[['age','bmi','children','sex','smoker']]),columns=['age','bmi','children','sex','smoker'],index=X_train_with.index) 
```


```python
# Combining the scaled numerical and categorical features using concat function
combined_data = pd.concat([scaled_data,X_train_with],axis =1)
combined_data = combined_data.drop(columns = 'sex')
print(combined_data.head(2))
```

              age       bmi  children    smoker  region_northeast  \
    479 -1.131915  0.311332 -0.918093 -0.502919             False   
    703 -0.351901 -0.699733 -0.102857 -0.502919             False   
    
         region_northwest  region_southeast  region_southwest  
    479             False              True             False  
    703              True             False             False  


### Training set model


```python
# Train a Linear Regression model
model1 = LinearRegression()
model1.fit(combined_data ,y_train)

# Intercept
intercept = model1.intercept_
print('Intercept:',intercept)

#Co-efficients
coefficient = model1.coef_
print('Coefficients are:',coefficient)
```

    Intercept: 13.106554036316927
    Coefficients are: [ 0.70143374  0.12574649  0.19479486  0.89856083  0.14293853  0.06049844
     -0.13476053 -0.06867643]



```python
### Checking the residuals r for:
## They have zero mean: E[ϵ]=0.
## The variance of these errors is constant (i.e., do not vary with time or value of any predictor): Var[ϵ]=σ2
```


```python
# Use the model to make predictions on the test data
predictions1 = model1.predict(combined_data)
```


```python
residuals1 =  y_train - predictions1  # Actual and predicted difference of values
print(residuals1.head(2))
```

    479   -0.753120
    703    0.034532
    Name: charges, dtype: float64



```python
residual_mean1 = residuals1.mean()
print('Residual mean:',residual_mean1)
```

    Residual mean: -1.7265524420338883e-16



```python
from sklearn.metrics import r2_score
R2 = r2_score(y_train, predictions1)
print('R2 score:',R2)
```

    R2 score: 0.7742718859387182



```python
training_mse1 = mean_squared_error(y_train, predictions1)
print('Mean squared error:',training_mse1)

training_rmse1 = mean_squared_error(y_train, predictions1,squared=False)
print('Root Mean squared error:',training_rmse1)
```

    Mean squared error: 0.39999032434042797
    Root Mean squared error: 0.6324478827068899


    /Users/nandhinimuthalraj/Documents/anaconda/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(



```python
## Looking for a even band of dots around zero
# Create a scatter plot
plt.scatter(predictions1,residuals1)
plt.title('Residual vs Predictions')
plt.xlabel('predictions')
plt.ylabel('residuals')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Categorical%20features.png)
    


### Test set model


```python
# Creating dummy variable for the categorical feature region
X_test_with = pd.get_dummies(X_test[['region']])

# Scaling the numerical features using standard scaler
from sklearn.preprocessing import StandardScaler

def scaler(df):
    scal = StandardScaler()
    scal.fit(df)
    scal_x = scal.transform(df)
    return scal_x

scaled_test_data = pd.DataFrame(scaler(X_test[['age','bmi','children','sex','smoker']]),columns=['age','bmi','children','sex','smoker'],index=X_test_with.index) 
```


```python
# Combining the scaled numerical and categorical features using concat function
combined_test_data = pd.concat([scaled_test_data,X_test_with],axis =1)

#Dropping the sex feature for better prediction
combined_test_data = combined_test_data.drop(columns = 'sex')
print(combined_test_data.head(2))
```

              age       bmi  children    smoker  region_northeast  \
    53  -0.303783  0.614478 -0.877258  1.902811             False   
    585 -0.521661 -0.387413  0.026993 -0.525538             False   
    
         region_northwest  region_southeast  region_southwest  
    53              False              True             False  
    585             False              True             False  



```python
# Train a Linear Regression model 
model1 = LinearRegression()
model1.fit(combined_test_data ,y_test)

# Intercept
intercept = model1.intercept_
print('Intercept:',intercept)

#Co-efficients
coefficient = model1.coef_
print('Coefficients are:',coefficient)
```

    Intercept: 13.213489361683923
    Coefficients are: [ 0.7001815   0.07681686  0.10216035  0.91680607  0.06265315 -0.05329095
      0.014536   -0.0238982 ]



```python
# Use the model to make predictions on the test data
predictions1 = model1.predict(combined_test_data)
```


```python
residuals1 =  y_test - predictions1  # Actual and predicted difference of values
print(residuals1.head(2))
```

    53      0.486493
    585    -0.131274
    1127   -0.131966
    228    -0.005787
    567     0.010805
              ...   
    503     1.189583
    1216   -0.174753
    591     0.151641
    132    -0.073739
    371     0.099604
    Name: charges, Length: 268, dtype: float64



```python
residual_mean1 = residuals1.mean()
print(residual_mean1)
```

    -8.285246452426542e-16



```python
from sklearn.metrics import r2_score
R2 = r2_score(y_test, predictions1)
print('R2 score:',R2)
```

    R2 score: 0.7409985126172374



```python
test_mse1 = mean_squared_error(y_test, predictions1)
print('Mean squared error:',test_mse1)

test_rmse1 = mean_squared_error(y_test, predictions1,squared=False)
print('Root Mean squared error:',test_rmse1)
```

    Mean squared error: 0.43919547868320324
    Root Mean squared error: 0.6627182498492125


    /Users/nandhinimuthalraj/Documents/anaconda/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
      warnings.warn(



```python
## Looking for a even band of dots around zero
# Create a scatter plot
plt.scatter(predictions1,residuals1)
plt.title('Residual vs Predictions')
plt.xlabel('predictions')
plt.ylabel('residuals')
plt.show()
```


    
![png](https://github.com/Muthaln1/Insurance-cost-analysis/blob/main/Test%20set%20model.png)
    


### Conclusion:
The training set with numerical and binary values has an R² score of -0.297, meaning that only 29.7% of the variance in the target variable (y) can be explained by the model using the independent variables (`age`, `bmi`, `children`, `smoker`). The mean squared error between the predictions and training values is 1.210, indicating a non-linear relationship with the outcome.

The training set with numerical and categorical values has an R² score of 0.7742, meaning that 77.42% of the variance in the target variable (y) can be explained by the model using the independent variables (`age`, `bmi`, `children`, `smoker`, `region_northwest`, `region_northeast`, `region_southeast`, `region_southwest`). The mean squared error between the predictions and training values is 0.399, suggesting a non-linear relationship with the outcome.

The test set with numerical and categorical values has an R² score of 0.74099, meaning that 74.09% of the variance in the target variable (y) can be explained by the model using the independent variables (`age`, `bmi`, `children`, `smoker`, `region_northwest`, `region_northeast`, `region_southeast`, `region_southwest`). The mean squared error between the predictions and test values is 0.43919, indicating a non-linear relationship with the outcome.

The graph of predictions vs. residuals is expected to form a band around zero, suggesting that the residuals should be near zero for accurate predictions. However, the above R² scores and mean squared errors indicate a non-linear relationship, which could potentially be addressed using more advanced models
