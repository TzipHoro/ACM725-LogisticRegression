import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from associated_risk import X, y


def roc_plot(actual: pd.Series, pred: pd.DataFrame, fill: str):
    import plotnine as gg
    tpr = lambda tp, p: tp / p
    fpr = lambda fp, n: fp / n

    x = dict()
    y = dict()

    for col in pred.columns:
        conf = pd.crosstab(actual, pred[col], rownames=['Actual'], colnames=['Predicted'], margins=True)
        TPR = tpr(conf.loc[1, 1], conf.loc[1, 'All'])
        FPR = fpr(conf.loc[0, 1], conf.loc[0, 'All'])

        x[col] = TPR
        y[col] = FPR

    roc = pd.DataFrame({'tpr': x, 'fpr': y})

    plot = (
        gg.ggplot(gg.aes(x=roc['fpr'], y=roc['tpr']))
        + gg.geom_line()
        + gg.geom_abline(intercept=0, slope=1)
        + gg.scale_x_continuous(limits=[0, 1])
        + gg.scale_y_continuous(limits=[0, 1])
        + gg.geom_area(fill=fill, alpha=0.4, linetype='solid')
    )

    return roc, plot


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
odds_ratios.to_latex('src/odds-ratios.tex')

# marginal effects
ame = log_reg.get_margeff(at='overall', method='dydx')
with open('src/marginal-effects.tex', 'w') as f:
    f.write(ame.summary().as_latex())

print(log_reg.summary())
print(ame.summary())

# determine threshold
yhat_in_sample = log_reg.predict(X_train)
pred_in_sample = pd.DataFrame()

for i in range(0, 100, 5):
    threshold = i / 100
    predicted = yhat_in_sample >= threshold
    pred_in_sample[threshold] = predicted.astype(int)

r, pl = roc_plot(y_train, pred_in_sample, fill='red')
pl.save('plots/roc-is.png')

threshold_corr = pred_in_sample.corrwith(y_train)
p_th = threshold_corr.idxmax()

# predict
yhat_out_of_sample = log_reg.predict(X_test)
pred_out_of_sample = pd.DataFrame()
for i in range(0, 100, 5):
    threshold = i / 100
    predicted = yhat_out_of_sample >= threshold
    pred_out_of_sample[threshold] = predicted.astype(int)
yhat = (yhat_out_of_sample >= p_th).astype(int)


r, pl = roc_plot(y_test, pred_out_of_sample, fill='blue')
pl.save('plots/roc-oos.png')

# score
conf_matrix = pd.crosstab(y_test, yhat, rownames=['Actual'], colnames=['Predicted'], margins=True)
accuracy = accuracy_score(y_test, yhat)
roc = classification_report(y_test, yhat, output_dict=True)
print(roc)
