import warnings
import pandas as pd
from scipy.stats.contingency import relative_risk

from data_summary import X, y

warnings.simplefilter(action='ignore', category=FutureWarning)

# create dummies for categorical vars
X['thal'] = X['thal'].astype('category')
X['cp'] = X['cp'].astype('category')
X['restecg'] = X['restecg'].astype('category')
X['slope'] = X['slope'].astype('category')
X['ca'] = X['ca'].astype('category')
X = pd.get_dummies(X, drop_first=True)
X = pd.get_dummies(X, drop_first=True)

# individual cross-tabs
df = X.copy()
df['age'] = (X['age'] > 54).astype(int)
df['trestbps'] = (df['trestbps'] > 130).astype(int)
df['chol'] = (df['chol'] > 250).astype(int)
df['thalach'] = (df['thalach'] > 150).astype(int)
df['oldpeak'] = (df['oldpeak'] > 1.1).astype(int)

# calculate ARD and RR
ard = lambda de, du, he, hu: round((de / (de + he)) - (du / (du + hu)), 4)
risk_diff = dict()
risk_ratio = dict()
ci = dict()

for col in df.columns:
    crosstab = pd.crosstab(df[col], y)
    De = crosstab.iloc[1, 1]
    Du = crosstab.iloc[0, 1]
    He = crosstab.iloc[1, 0]
    Hu = crosstab.iloc[0, 0]

    risk_diff[col] = ard(De, Du, He, Hu)
    rr = relative_risk(De, De + He, Du, Du + Hu)
    risk_ratio[col] = round(rr.relative_risk, 4)
    ci[col] = rr.confidence_interval(confidence_level=0.95)


risk = pd.DataFrame.from_records([risk_diff, risk_ratio, ci],
                                 index=['Risk Difference', 'Risk Ratio', 'RR 95% Confidence Interval'])
risk = risk.transpose()
risk['RR 95% Confidence Interval'] = risk['RR 95% Confidence Interval'].apply(
    lambda row: tuple([round(i, 4) for i in row]))
risk['RR 95% Confidence Interval'] = risk['RR 95% Confidence Interval'].apply(
    lambda row: str(row) + '*' if not (row[0] <= 1 <= row[1]) else str(row))

risk.set_axis(['age', 'sex', 'resting blood pressure > 130', 'serum cholesterol > 250 ml/dl',
               'fasting blood sugar > 120 mg/dl', 'maximum heart rate achieved > 150', 'exercise induced angina',
               'oldpeak', 'chest pain = 1', 'chest pain = 2', 'chest pain = 3', 'resting electrocardiograph = 1',
               'resting electrocardiograph = 2', 'slope = 1', 'slope = 2', 'major vessels colored = 1',
               'major vessels colored = 2', 'major vessels colored = 3', 'major vessels colored = 4',
               'thalassemia = 1', 'thalassemia = 2', 'thalassemia = 3'], inplace=True)

print('RD:')
print(risk[risk['Risk Difference'] == risk['Risk Difference'].max()])
print(risk[risk['Risk Difference'] == risk['Risk Difference'].min()])
print('RR:')
print(risk[risk['Risk Ratio'] == risk['Risk Ratio'].max()])
print(risk[risk['Risk Ratio'] == risk['Risk Ratio'].min()])

risk.to_latex('src/risk-tables.tex', column_format='lrrr')
