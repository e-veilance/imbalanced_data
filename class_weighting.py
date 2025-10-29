import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

# separate features and target
y = df['stroke']
X = df.drop(columns=['id','stroke'])
#print(y.head)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Compute class weights
# weighting minority class more heavily to counteract imbalanced dataset
# by increasing loss for misclassifications of minority class data
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# fit logistic regression models with and without weighted classes
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

lr_classifier_weighted = LogisticRegression(class_weight=class_weight_dict)
lr_classifier_weighted.fit(X_train, y_train)

# Evaluate models
# weighting the classes produces a model with fewer false negatives
# though in practice other methods should be used in combination
print(f'Model    | Accuracy | F1 Score | AUC')

base_predicted = lr_classifier.predict(X_test)
base_accuracy = accuracy_score(y_test, base_predicted)
base_f1 = f1_score(y_test, base_predicted)
base_auc = roc_auc_score(y_test, base_predicted)
print(f'Base     |  {base_accuracy*100:.2f}%  |  {base_f1*100:.2f}%  | {base_auc*100:.2f}%')

weighted_predicted = lr_classifier_weighted.predict(X_test)
weighted_accuracy = accuracy_score(y_test, weighted_predicted)
weighted_f1 = f1_score(y_test, weighted_predicted)
weighted_auc = roc_auc_score(y_test, weighted_predicted)
print(f'Weighted |  {weighted_accuracy*100:.2f}%  |  {weighted_f1*100:.2f}%  | {weighted_auc*100:.2f}%')