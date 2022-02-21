import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
from sklearn.impute import SimpleImputer

# read excel, downloading package openpyxl is needed
data = pd.read_excel('diplo_data_final_3v.xlsx', engine='openpyxl')

# creating dataframe from columns, that will be used for simple regression analysis - without division into areas
df = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony",
                                 "bus", "cellar", "floor", "garage", "lift", "pharmacy",
                                 "post", "price_per_month", "restaurant", "school", "shop",
                                 "sport_field", "train", "tram", "underground"])

# creating dataframe of predictors - independent variables
df_predictors = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony", "bus", "cellar", "floor", "garage",
                                            "lift", "pharmacy", "post", "restaurant", "school", "shop", "sport_field",
                                            "train", "tram", "underground"])

# creating dataframe of dependent variable
df_dependent_variable = pd.DataFrame(data, columns=["price_per_month"])

# checking if any of the columns contains missing values - our dataset has 0 missing values
for i in df_predictors.columns:
    print(df_predictors[i].isnull().value_counts())

# checking of types of independent variables used in models
print(df_predictors.dtypes)

# in case of missing values, these code fill none with mean of values, modifying according to used dataset is needed
# there are other strategies to filling none values as well

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(df_predictors.iloc[:, 2:3])
# df_predictors.iloc[:, 2:3] = imputer.transform(df_predictors.iloc[:, 2:3])
#
# imputer.fit(df_predictors.iloc[:, 4:5])
# df_predictors.iloc[:, 4:5] = imputer.transform(df_predictors.iloc[:, 4:5])
#
# imputer.fit(df_predictors.iloc[:, 9:17])
# df_predictors.iloc[:, 9:17] = imputer.transform(df_predictors.iloc[:, 9:17])
#
# df_predictors['accommodation'] = df_predictors['accommodation'].fillna(False)
# df_predictors['balcony'] = df_predictors['balcony'].fillna(False)
# df_predictors['cellar'] = df_predictors['cellar'].fillna(False)
# df_predictors['garage'] = df_predictors['garage'].fillna(False)
# df_predictors['lift'] = df_predictors['lift'].fillna(False)

# cast of variables if needed
# df_predictors = np.asarray(df_predictors).astype('int')
# df_dependent_variable = np.asarray(df_dependent_variable).astype('int')
# X = df_predictors.var()
# y = df_dependent_variable("price_per_month")

# adding constant to a model - column with only ones
df_predictors = sm.add_constant(df_predictors)

# estimation of the first model, predictors are variables describing flat and distance from flat and points of interest
model = sm.OLS(df_dependent_variable, df_predictors).fit()
predictions = model.predict(df_predictors)

# print out the statistics
print(model.summary())

# from previous model we can see, that floor variable is not significant, so there is a estimation of the
# model without floor in order to improve significance of model
df_predictors_2 = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony", "bus", "cellar", "garage", "lift",
                                              "pharmacy", "post", "restaurant", "school", "shop", "sport_field",
                                              "train", "tram", "underground"])
# casting if needed
# df_predictors_2 = np.asarray(df_predictors_2).astype('int')
# df_dependent_variable = np.asarray(df_dependent_variable).astype('int')

df_predictors_2 = sm.add_constant(df_predictors_2)

model_2 = sm.OLS(df_dependent_variable, df_predictors_2).fit()
predictions_2 = model_2.predict(df_predictors_2)

# print out the statistics
print(model_2.summary())

# we add first type of location, we actually have 22 areas, but locations are dummy variables, so we have to use less
# locations in order to prevent perfect multicollinearity from appearing in model

df_predictors_3 = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony", "bus", "cellar", "floor", "garage",
                                              "lift",
                                              "pharmacy",
                                              "post", "restaurant", "school", "shop", "sport_field", "train", "tram",
                                              "underground", "oblast_2", "oblast_3", "oblast_4", "oblast_5", "oblast_6",
                                              "oblast_7",
                                              "oblast_8", "oblast_9", "oblast_10", "oblast_11", "oblast_12",
                                              "oblast_13", "oblast_14",
                                              "oblast_15", "oblast_16", "oblast_17", "oblast_18", "oblast_19",
                                              "oblast_20", "oblast_21"])

df_predictors_3 = sm.add_constant(df_predictors_3)

model_3 = sm.OLS(df_dependent_variable, df_predictors_3).fit()
predictions_3 = model_3.predict(df_predictors_3)

# print out the statistics
print(model_3.summary())

# model with only significant variables and locations -> improvement
df_predictors_4 = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony", "bus", "cellar", "floor", "garage",
                                              "lift",
                                              "pharmacy",
                                              "restaurant", "shop", "tram",
                                              "underground",
                                              "oblast_3", "oblast_4", "oblast_5", "oblast_6",
                                              "oblast_7",
                                              "oblast_8", "oblast_9", "oblast_10", "oblast_11", "oblast_12",
                                              "oblast_13", "oblast_14",
                                              "oblast_15", "oblast_16", "oblast_17", "oblast_18", "oblast_19",
                                              "oblast_20", "oblast_21"])

df_predictors_4 = sm.add_constant(df_predictors_4)

model_4 = sm.OLS(df_dependent_variable, df_predictors_4).fit()
predictions_4 = model_4.predict(df_predictors_4)

# print out the statistics
print(model_4.summary())

# Jarque-Bera test for normality of residuals, H0: Normality of residuals, H1: Non-normality of residuals
print("-------")
print(stats.jarque_bera(df_predictors_3))
print(stats.jarque_bera(df_predictors_4))

# robust linear regression model, data are the same as used in the third model
df_predictors_5 = pd.DataFrame(data, columns=["equipment", "area", "atm", "balcony", "bus", "cellar", "floor", "garage",
                                              "lift",
                                              "pharmacy",
                                              "post", "restaurant", "school", "shop", "sport_field", "train", "tram",
                                              "underground", "oblast_2", "oblast_3", "oblast_4", "oblast_5", "oblast_6",
                                              "oblast_7",
                                              "oblast_8", "oblast_9", "oblast_10", "oblast_11", "oblast_12",
                                              "oblast_13", "oblast_14",
                                              "oblast_15", "oblast_16", "oblast_17", "oblast_18", "oblast_19",
                                              "oblast_20", "oblast_21"])

df_predictors_5 = sm.add_constant(df_predictors_5)

model_5 = sm.RLM(df_dependent_variable, df_predictors_5).fit()
predictions_5 = model_5.predict(df_predictors_5)

# print out the statistics
print(model_5.summary())
