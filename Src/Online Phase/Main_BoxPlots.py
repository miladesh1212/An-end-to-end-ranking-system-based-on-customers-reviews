# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:54:36 2019

@author: Milad
"""

## Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############# BoxPlot #######################

df = pd.read_excel('Scores2.xls')
sns.boxplot(x=df)
plt.xlim([0, 3])
plt.show()