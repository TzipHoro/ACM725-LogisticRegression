"""
data_summary.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimpy import skim


# read data
df = pd.read_csv('data/heart.csv')
skim(df)

X = df.drop('target', axis=1)
y = df['target']

corr = df.corr()


def get_lower_tri_heatmap(data, output="src/plots/correlation.png"):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(data, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output)


if __name__ == '__main__':
    import plotnine as gg

    # plot all sub-distributions
    plots = []
    for col in X:
        plot = gg.ggplot(df) +\
               gg.geom_boxplot(gg.aes(y=col, x='factor(target)', color='factor(target)'), notch=True) +\
               gg.xlab('target') +\
               gg.coord_flip() +\
               gg.scale_color_discrete(guide=False) +\
               gg.theme(text=gg.element_text(size=24))
        plot.save(f'src/plots/target-{col}.png', 'png')

    # corr plot
    get_lower_tri_heatmap(corr)

