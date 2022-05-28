import pandas as pd

data1 = pd.read_excel("D:\project\sampla_data_08_05_2022(final).xlsx")

# Dummy variables
data1.head()
data1.info()

data1= data1.drop(["Patient_ID"], axis=1)
data1= data1.drop(["Patient_Age"], axis=1)
data1= data1.drop(["Test_Booking_Date"], axis=1)
data1= data1.drop(["Sample_Collection_Date"], axis=1)
data1= data1.drop(["Mode_Of_Transport"], axis=1)
data1.columns


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["Test_Name"] = le.fit_transform(data1["Test_Name"])
data1["Sample"] = le.fit_transform(data1["Sample"])
data1["Way_Of_Storage_Of_Sample"] = le.fit_transform(data1["Way_Of_Storage_Of_Sample"])
data1["Cut-off Schedule"] = le.fit_transform(data1["Cut-off Schedule"])
data1["Traffic_Conditions"] = le.fit_transform(data1["Traffic_Conditions"])
data1["Reached_On_Time"] = le.fit_transform(data1["Reached_On_Time"])
data1["Patient_Gender"] = le.fit_transform(data1["Patient_Gender"])

data1.columns
#or

# n-1 dummy variables will be created for n categories
data1 = pd.get_dummies(df, columns = ["Patient_Gender", "Test_Name", "Sample", "Way_Of_Storage_Of_Sample", "Test_Booking_Date", "Sample_Collection_Date", "Cut-off Schedule", "Traffic_Conditions", "Mode_Of_Transport", "Reached_On_Time"], drop_first = True)

data1.head()


# Input and Output Split
predictors = data1.loc[:, data1.columns!="Reached_On_Time"]
type(predictors)

target = data1["Reached_On_Time"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(x_train))
accuracy_score(y_train, rf_clf.predict(x_train))



######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(x_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))


import pickle
pickle.dump(rf_clf, open('Medical_sample.pickle', 'wb'))
#load the model from disk
model = pickle.load(open('Medical_sample.pickle', 'rb'))

# checking for the results
list_value = pd.DataFrame(data1.iloc[0:1,:14])
list_value

print(cv_rf_clf_grid.predict(list_value))


import pickle
filename = "rfd.pkl"
pickle.dump(rf_clf, open(filename,"wb"))

filename1 = "final.pkl"
pickle.dump(data1, open(filename1,"wb"))


