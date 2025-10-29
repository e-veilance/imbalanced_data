import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
# random forests implement bagging by default, which should help prevent
# overfitting on imbalanced data
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

rf_classifier_smote = RandomForestClassifier(n_estimators=100)
rf_classifier_smote.fit(X_train_smote, y_train_smote)

rf_classifier_rus = RandomForestClassifier(n_estimators=100)
rf_classifier_rus.fit(X_train_rus, y_train_rus)

# with balanced_subsample, weights are adjusted inservely proportional to the
# target class frequency in the bootstrap sample per tree
rf_classifier_b = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
rf_classifier_b.fit(X_train, y_train)

rf_classifier_b_smote = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
rf_classifier_b_smote.fit(X_train_smote, y_train_smote)

rf_classifier_b_rus = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
rf_classifier_b_rus.fit(X_train_rus, y_train_rus)

# fit gradient boosting classifier to each dataset
# models are trained an data sequentially, allowing learning from previous model
# misclassifications of the minority class
gb_classifier = GradientBoostingClassifier(n_estimators=100)
gb_classifier.fit(X_train, y_train)

gb_classifier_smote = GradientBoostingClassifier(n_estimators=100)
gb_classifier_smote.fit(X_train_smote, y_train_smote)

gb_classifier_rus = GradientBoostingClassifier(n_estimators=100)
gb_classifier_rus.fit(X_train_rus, y_train_rus)

# score each classifier
# base dataset: high accuracy, low f1 score due to false negatives
# SMOTE dataset: high accuracy and f1 score
# RUS: middling accuracy and f1 score due to high information loss due to the
# severity of imbalace in the dataset
# 
# oversampling minority class with SMOTE seems to be a better
# choice for datasets as imbalanced as these (though combining under and over
# sampling may be the best decision in general)
#
# bagging via RF seems to be a better solution for this dataset compared to
# gradient boosting; using balanced_subsample does not seem to significantly
# impact results
rf_predicted = rf_classifier.predict(X_test)
base_accuracy = accuracy_score(y_test, rf_predicted)
base_f1 = f1_score(y_test, rf_predicted)
base_auc = roc_auc_score(y_test, rf_predicted)
print('RF Base Dataset  Accuracy, F1 Score, AUC')
print(f'                 {base_accuracy*100:.2f}%    {base_f1*100:.2f}%    {base_auc*100:.2f}%')

smote_predicted = rf_classifier_smote.predict(X_test_smote)
smote_accuracy = accuracy_score(y_test_smote, smote_predicted)
smote_f1 = f1_score(y_test_smote, smote_predicted)
smote_auc = roc_auc_score(y_test_smote, smote_predicted)
print('RF SMOTE Dataset Accuracy, F1 Score, AUC')
print(f'                 {smote_accuracy*100:.2f}%    {smote_f1*100:.2f}%    {smote_auc*100:.2f}%')

rus_predicted = rf_classifier_rus.predict(X_test_rus)
rus_accuracy = accuracy_score(y_test_rus, rus_predicted)
rus_f1 = f1_score(y_test_rus, rus_predicted)
rus_auc = roc_auc_score(y_test_rus, rus_predicted)
print('RF RUS Dataset   Accuracy, F1 Score, AUC')
print(f'                 {rus_accuracy*100:.2f}%    {rus_f1*100:.2f}%    {rus_auc*100:.2f}%')

b_predicted = rf_classifier_b.predict(X_test)
b_accuracy = accuracy_score(y_test, b_predicted)
b_f1 = f1_score(y_test, b_predicted)
b_auc = roc_auc_score(y_test, b_predicted)
print('RF Base Dataset w/ Balanced Subsampling  Accuracy, F1 Score, AUC')
print(f'                 {b_accuracy*100:.2f}%    {b_f1*100:.2f}%    {b_auc*100:.2f}%')

b_smote_predicted = rf_classifier_b_smote.predict(X_test_smote)
b_smote_accuracy = accuracy_score(y_test_smote, b_smote_predicted)
b_smote_f1 = f1_score(y_test_smote, b_smote_predicted)
b_smote_auc = roc_auc_score(y_test_smote, b_smote_predicted)
print('RF SMOTE Dataset w/ Balanced Subsampling Accuracy, F1 Score, AUC')
print(f'                 {b_smote_accuracy*100:.2f}%    {b_smote_f1*100:.2f}%    {b_smote_auc*100:.2f}%')

b_rus_predicted = rf_classifier_b_rus.predict(X_test_rus)
b_rus_accuracy = accuracy_score(y_test_rus, b_rus_predicted)
b_rus_f1 = f1_score(y_test_rus, b_rus_predicted)
b_rus_auc = roc_auc_score(y_test_rus, b_rus_predicted)
print('RF RUS Dataset w/ Balanced Subsampling   Accuracy, F1 Score, AUC')
print(f'                 {b_rus_accuracy*100:.2f}%    {b_rus_f1*100:.2f}%    {b_rus_auc*100:.2f}%')

gb_base_predicted = gb_classifier.predict(X_test)
gb_base_accuracy = accuracy_score(y_test, gb_base_predicted)
gb_base_f1 = f1_score(y_test, gb_base_predicted)
gb_base_auc = roc_auc_score(y_test, gb_base_predicted)
print('GB Base Dataset  Accuracy, F1 Score, AUC')
print(f'                 {gb_base_accuracy*100:.2f}%    {gb_base_f1*100:.2f}%    {gb_base_auc*100:.2f}%')

gb_smote_predicted = gb_classifier_smote.predict(X_test_smote)
gb_smote_accuracy = accuracy_score(y_test_smote, gb_smote_predicted)
gb_smote_f1 = f1_score(y_test_smote, gb_smote_predicted)
gb_smote_auc = roc_auc_score(y_test_smote, gb_smote_predicted)
print('GB SMOTE Dataset Accuracy, F1 Score, AUC')
print(f'                 {gb_smote_accuracy*100:.2f}%    {gb_smote_f1*100:.2f}%    {gb_smote_auc*100:.2f}%')

gb_rus_predicted = gb_classifier_rus.predict(X_test_rus)
gb_rus_accuracy = accuracy_score(y_test_rus, gb_rus_predicted)
gb_rus_f1 = f1_score(y_test_rus, gb_rus_predicted)
gb_rus_auc = roc_auc_score(y_test_rus, gb_rus_predicted)
print('GB RUS Dataset   Accuracy, F1 Score, AUC')
print(f'                 {gb_rus_accuracy*100:.2f}%    {gb_rus_f1*100:.2f}%    {gb_rus_auc*100:.2f}%')