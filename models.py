import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

from associated_risk import X, y

# import ROCMetrics
sys.path.insert(0, r'C:\Users\tzipo\Documents\GitHub\NeuralNetworks-FinalProject')
from ROC import ROCMetrics

plt.style.use('ggplot')


# train/test split
X.columns = ['age', 'sex', 'resting blood pressure', 'serum cholesterol', 'fasting blood sugar > 120 mg/dl', 
             'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'chest pain = 1', 'chest pain = 2', 
             'chest pain = 3', 'resting electrocardiograph = 1', 'resting electrocardiograph = 2', 'slope = 1', 
             'slope = 2', 'major vessels colored = 1', 'major vessels colored = 2', 'major vessels colored = 3',
             'major vessels colored = 4', 'thalassemia = 1', 'thalassemia = 2', 'thalassemia = 3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# logistic regression
log_reg = sm.Logit(y_train, X_train).fit()
with open('src/logistic-regression.tex', 'w') as f:
    f.write(log_reg.summary().as_latex())

# calculate odds ratios:
odds_ratios = pd.DataFrame(
    {
        "Odds Ratio": log_reg.params,
        "Lower CI": log_reg.conf_int()[0],
        "Upper CI": log_reg.conf_int()[1],
    }
)
odds_ratios = np.exp(odds_ratios)
odds_ratios['OR 95% Confidence Interval'] = odds_ratios.apply(lambda row:
                                                              (round(row.iloc[1], 4), round(row.iloc[2], 4)),
                                                              axis=1)
odds_ratios.drop(["Lower CI", "Upper CI"], inplace=True, axis=1)
odds_ratios['Odds Ratio'] = odds_ratios['Odds Ratio'].round(4)
odds_ratios['OR 95% Confidence Interval'] = odds_ratios['OR 95% Confidence Interval'].apply(
    lambda row: str(row) + '*' if not (row[0] <= 1 <= row[1]) else str(row))
print(odds_ratios[odds_ratios['Odds Ratio'] == odds_ratios['Odds Ratio'].max()])
print(odds_ratios[odds_ratios['Odds Ratio'] == odds_ratios['Odds Ratio'].min()])
odds_ratios.to_latex('src/odds-ratios.tex')

# marginal effects
ame = log_reg.get_margeff(at='overall', method='dydx')
with open('src/marginal-effects.tex', 'w') as f:
    f.write(ame.summary().as_latex())

ame_df = ame.summary_frame()
print(ame_df[ame_df['dy/dx'] == ame_df['dy/dx'].max()])
print(ame_df[ame_df['dy/dx'] == ame_df['dy/dx'].min()])

# determine threshold
yhat_in_sample = log_reg.predict(X_train)
roc = ROCMetrics(y_train, yhat_in_sample)
thresholds = roc.threshold_matrix(step_size=0.001)
p_th = thresholds.loc[['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score'], :].sum().idxmax()
print(thresholds.loc[:, p_th].round(4))
roc.roc_plot(thresholds, 'plots/roc-is-1.png')

# scoring in sample
brier_score = brier_score_loss(y_train, yhat_in_sample)
auc_ = roc_auc_score(y_train, yhat_in_sample)
ll = log_loss(y_train, yhat_in_sample)

# predict
yhat_out_of_sample = log_reg.predict(X_test)
roc_oos = ROCMetrics(y_test, yhat_out_of_sample)
thresholds_oos = roc_oos.threshold_matrix(step_size=0.001)
p_th = thresholds_oos.loc[['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score'], :].sum().idxmax()
print(thresholds.loc[:, p_th].round(4))
roc_oos.roc_plot(thresholds, 'plots/roc-oos-1.png')

# scoring oos
brier_score_oos = brier_score_loss(y_test, yhat_out_of_sample)
auc_oos = roc_auc_score(y_test, yhat_out_of_sample)
ll_oos = log_loss(y_test, yhat_out_of_sample)
