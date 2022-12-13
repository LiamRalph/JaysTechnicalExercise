import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRFClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
import numpy as np




# Load Training Data
df =  pd.read_csv('./data/training.csv')


#Seperate pitch information and outcome
Target = df.InPlay
df.drop(columns=['InPlay'], inplace=True)
#Split data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(df, Target, random_state=37, test_size=0.2)





#Fit the model

params = {
    'n_estimators': 50,
    'max_depth': 2
}
XGB = XGBClassifier(**params, objective= 'binary:logistic', seed=37)
XGB.fit(X_train, y_train)



#Hyperparameter Tuning

#Random Forest
#{'max_depth': 2, 'n_estimators': 50}
# XGB = XGBRFClassifier(objective= 'binary:logistic', seed=37)
# test_params = {
#     'n_estimators':[50, 65, 80, 100, 115, 130, 150, 200],
#     'max_depth':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# }
 
# XGB = GridSearchCV(estimator = XGB, param_grid = test_params)
# XGB.fit(X_train, y_train)
# print(XGB.best_params_)

#Gradient Boosted
#{'max_depth': 2, 'n_estimators': 50}
# XGB = XGBClassifier(objective= 'binary:logistic', seed=37)
# test_params = {
#    'n_estimators':[50, 65, 80, 100, 115, 130, 150, 200],
#    'max_depth':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# }

# XGB = GridSearchCV(estimator = XGB, param_grid = test_params)
# XGB.fit(X_train, y_train)
# print(XGB.best_params_)






#Save model for future use
dump(XGB,'BallsInPlayXGB') 

#Print Model Accuracy
print("Model Accuracy: %.3f" % XGB.score(X_test, y_test))


# Plot feature importance
plot_importance(XGB, importance_type='gain')
plt.show()



#Load deployment data and predict each pitches probability of in play
x_deploy =  pd.read_csv('./data/deploy.csv') 
y_deploy = XGB.predict_proba(x_deploy)

#Load probability of in play and save to output csv
x_deploy["InPlayProbability"] = [value[1] for value in y_deploy]
x_deploy.to_csv("./data/output.csv", index=False)
