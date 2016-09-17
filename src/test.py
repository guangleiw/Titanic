#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from pandas import Series ,DataFrame

data_train = pd.read_csv("/Users/wangguanglei/dev/py_workspace/Titanic/data/train.csv")

##data_train.info()
print(data_train.describe())