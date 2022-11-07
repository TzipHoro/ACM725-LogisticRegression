import pandas as pd
import plotnine as gg
from skimpy import skim


# read data
df = pd.read_csv('data/heart.csv')
skim(df)

X = df.drop('target', axis=1)
y = df['target']

plots = []
for col in X:
    plot = gg.ggplot(df) +\
           gg.geom_boxplot(gg.aes(y=col, x='factor(target)', color='factor(target)'), notch=True) +\
           gg.xlab('target') +\
           gg.coord_flip() +\
           gg.scale_color_discrete(guide=False)
    plot.save(f'src/plots/target-{col}.png', 'png')