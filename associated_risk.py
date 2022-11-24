import pandas as pd
from scipy.stats.contingency import relative_risk

from data_summary import X, y

# individual cross-tabs
age = pd.crosstab((X['age'] > 54).astype(int), y)
sex = pd.crosstab(X['sex'], y)
cp = pd.crosstab(X['cp'].isin([1, 2]).astype(int), y)
trestbps = pd.crosstab((X['trestbps'] > 130).astype(int), y)
chol = pd.crosstab((X['chol'] > 250).astype(int), y)
fbs = pd.crosstab(X['fbs'], y)
restecg = pd.crosstab(X['restecg'].isin([0, 1]).astype(int), y)
thalach = pd.crosstab((X['thalach'] > 150).astype(int), y)
exang = pd.crosstab(X['exang'], y)
oldpeak = pd.crosstab((X['oldpeak'] > 1.1).astype(int), y)
slope = pd.crosstab(X['slope'].isin([1, 2]).astype(int), y)
ca = pd.crosstab(X['ca'].isin([0, 1, 2]).astype(int), y)
thal = pd.crosstab(X['thal'].isin([2, 3]).astype(int), y)

crosstabs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# calculate ARD and RR
ard = lambda de, du, he, hu: round((de / (de + he)) - (du / (du + hu)), 4)
# rr = lambda de, du, he, hu: round((de / (de + he)) / (du / (du + hu)), 4)
risk_diff = dict()
risk_ratio = dict()
ci = dict()

for df in crosstabs:
    De = df.iloc[1, 1]
    Du = df.iloc[0, 1]
    He = df.iloc[1, 0]
    Hu = df.iloc[0, 0]

    var = df.index.name
    risk_diff[var] = ard(De, Du, He, Hu)
    rr = relative_risk(De, De + He, Du, Du + Hu)
    risk_ratio[var] = round(rr.relative_risk, 4)
    ci[var] = rr.confidence_interval(confidence_level=0.95)

risk = pd.DataFrame.from_records([risk_diff, risk_ratio, ci],
                                 index=['Risk Difference', 'Risk Ratio', 'RR 95% Confidence Interval'])
risk = risk.transpose()
risk['RR 95% Confidence Interval'] = risk['RR 95% Confidence Interval'].apply(lambda row: tuple([round(i, 4) for i in row]))
# risk['Correlation'] = X.corrwith(y)
# risk = risk[['Correlation', 'Risk Difference', 'Risk Ratio']]
risk.set_axis(['age', 'sex', 'chest pain ∈ {1, 2}', 'resting blood pressure > 130', 'serum cholestoral > 250 ml/dl',
               'fasting blood sugar > 120 mg/dl', 'resting electrocardiographic results ∈ {0, 1}',
               'maximum heart rate achieved > 150', 'exercise induced angina', 'oldpeak', 'slope',
               'major vessels colored ∈ {0, 1, 2}', 'thal ∈ {2, 3}'], inplace=True)
risk.to_latex('src/risk-tables.tex', column_format='lrrr')
