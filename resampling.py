import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
print(y.head)

# Synthetic Minority Over-sampling Technique (SMOTE)
# constructs new samples of minority class based on current samples
# risk of overfitting on available minority sample data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print(y_smote.head)

# Random Under Sampling
# randomly discord samples of the majority class
# risk of information loss
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print(y_rus.head)