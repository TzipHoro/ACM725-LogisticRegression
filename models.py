import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss, roc_auc_score, log_loss, \
    roc_curve, auc, RocCurveDisplay, confusion_matrix

from associated_risk import X, y

plt.style.use('ggplot')


def specificity(tn, fp):
    return tn / (tn + fp)


def sensitivity(tp, fn):
    return tp / (tp + fn)


def precision(tp, fp):
    return tp / (tp + fp)


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


# train/test split
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
pred_in_sample = pd.DataFrame()

sensitivity_is = [None] * 100
specificity_is = [None] * 100
precision_is = [None] * 100
accuracy_is = [None] * 100

for i in range(0, 100, 1):
    threshold = i / 100
    predicted = yhat_in_sample >= threshold
    pred_in_sample[threshold] = predicted.astype(int)

    conf_matrix = confusion_matrix(y_train, predicted)
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]

    sensitivity_is[i] = sensitivity(tp, fn)
    specificity_is[i] = specificity(tn, fp)
    precision_is[i] = precision(tp, fp)
    accuracy_is[i] = accuracy_score(y_train, predicted)

roc_df_is = pd.DataFrame({'sesitivity': sensitivity_is, 'specificity': specificity_is, 'precision': precision_is,
                          'accuracy': accuracy_is},
                         index=pred_in_sample.columns).transpose()
pred_in_sample = pred_in_sample.append(roc_df_is)
p_th = pred_in_sample.loc[['sesitivity', 'specificity', 'precision', 'accuracy'], :].sum().idxmax()

# scoring in sample
brier_score = brier_score_loss(y_train, yhat_in_sample)
auc_ = roc_auc_score(y_train, yhat_in_sample)
ll = log_loss(y_train, yhat_in_sample)
fpr, tpr, thresholds = roc_curve(y_train, yhat_in_sample)
roc_auc = auc(fpr, tpr)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='target')
display.plot()
plt.savefig('plots/roc-is-1.png')

# predict
yhat_out_of_sample = log_reg.predict(X_test)
pred_out_of_sample = pd.DataFrame()
for i in range(0, 100, 1):
    threshold = i / 100
    predicted = yhat_out_of_sample >= threshold
    pred_out_of_sample[threshold] = predicted.astype(int)
yhat = (yhat_out_of_sample >= p_th).astype(int)

# scoring oos
brier_score_oos = brier_score_loss(y_test, yhat_out_of_sample)
auc_oos = roc_auc_score(y_test, yhat_out_of_sample)
ll_oos = log_loss(y_test, yhat_out_of_sample)
fpr_oos, tpr_oos, thresholds_oos = roc_curve(y_test, yhat_out_of_sample)
roc_auc_oos = auc(fpr_oos, tpr_oos)

display = RocCurveDisplay(fpr=fpr_oos, tpr=tpr_oos, roc_auc=roc_auc_oos, estimator_name='target')
display.plot()
plt.savefig('plots/roc-oos-1.png')

# score
conf_matrix = pd.crosstab(y_test, yhat, rownames=['Actual'], colnames=['Predicted'], margins=True)
accuracy_ = accuracy_score(y_test, yhat)
roc = classification_report(y_test, yhat)
print(roc)
