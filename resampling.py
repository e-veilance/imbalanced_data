import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# read in dataset
df = pd.read_csv("cerebral_stroke_prediction.csv")

# drop columns with NaN values
df.drop(columns=['bmi', 'smoking_status'], inplace=True)

# convert categorical columns to numerical codes
df['gender'] = df['gender'].astype('category')
df['ever_married'] = df['ever_married'].astype('category')
df['work_type'] = df['work_type'].astype('category')
df['Residence_type'] = df['Residence_type'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

# seperate features and target
y = df['stroke']
X = df.drop(columns=['id','stroke'])
#print(y.head)

# Synthetic Minority Over-sampling Technique (SMOTE)
# constructs new samples of minority class based on current samples
# risk of overfitting on available minority sample data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
#print(y_smote.head)

# Random Under Sampling
# randomly discord samples of the majority class
# risk of information loss
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
#print(y_rus.head)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=.2, random_state=42)
X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, y_rus, test_size=.2, random_state=42)

# fit random forests to different datasets
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

rf_classifier_smote = RandomForestClassifier(n_estimators=100)
rf_classifier_smote.fit(X_train_smote, y_train_smote)

rf_classifier_rus = RandomForestClassifier(n_estimators=100)
rf_classifier_rus.fit(X_train_rus, y_train_rus)

# score each random forest
base_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
base_f1 = f1_score(y_test, rf_classifier.predict(X_test))
print('Base Dataset  Accuracy, F1 Score')
print(f'              {base_accuracy*100:.2f}%    {base_f1*100:.2f}%')
# high accuracy, but low f1 score due to false negatives

smote_accuracy = accuracy_score(y_test_smote, rf_classifier_smote.predict(X_test_smote))
smote_f1 = f1_score(y_test_smote, rf_classifier_smote.predict(X_test_smote))
print('SMOTE Dataset Accuracy, F1 Score')
print(f'              {smote_accuracy*100:.2f}%    {smote_f1*100:.2f}%')
# high accuracy and f1 score, oversampling minority class seems to be a better
# choice for datasets as imbalanced as these (though combining under and over
# sampling may be the best decision in general)

rus_accuracy = accuracy_score(y_test_rus, rf_classifier_rus.predict(X_test_rus))
rus_f1 = f1_score(y_test_rus, rf_classifier_rus.predict(X_test_rus))
print('RUS Dataset   Accuracy, F1 Score')
print(f'              {rus_accuracy*100:.2f}%    {rus_f1*100:.2f}%')
# middling accuracy and f1 score due to high information loss due to the severity
# of imbalace in the dataset