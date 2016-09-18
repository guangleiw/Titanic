#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas import Series ,DataFrame

data_train = pd.read_csv("/Users/wangguanglei/dev/py_workspace/Titanic/data/train.csv")


fig = plt.figure()
fig.set(alpha = 0.2) # 设定图表颜色alpha 参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
df.plot(kind='bar',stacked = True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df = pd.DataFrame({u"男性":Survived_m,u"女性":Survived_f})
df.plot(kind='bar',stacked = True)
plt.title(u"按性别看获救状况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()
