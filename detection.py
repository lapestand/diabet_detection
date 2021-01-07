import pandas as pd
import os
import numpy as np
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter


DATASET_FILE = "diabetes_data_upload.csv"



os.system("cls")

jvm.start()

loader = Loader(classname="weka.core.converters.CSVLoader")
data = loader.load_file(DATASET_FILE)
data.class_is_last()

print(data)

jvm.stop()
'''
df = pd.read_csv(DATASET_FILE, sep=";")

print(df)

jvm.start()
cls = Classifier(classname="weka.classifier.trees.J48")
print(cls)
jvm.stop()

'''