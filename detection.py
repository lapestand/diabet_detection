import pandas as pd
import os
import numpy as np
import random
import argparse
import logging

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
from weka.classifiers import Classifier, Evaluation

from weka.core.classes import Random

def classify(k):
    DATASET_FILE    = "diabetes_data_upload.csv"
    KNN             = "weka.classifiers.lazy.IBk"
    K_VALUE         = k


    # Clear screen
    os.system("cls")

    # Start JVM
    try:
        jvm.start()
    except RuntimeError:
        jvm.stop()
        jvm.start()

    # Create a Loader object
    loader = Loader(classname="weka.core.converters.CSVLoader")

    # Load the dataset
    data = loader.load_file(DATASET_FILE)

    # Indicates that class is the last attribute
    data.class_is_last()


    # Create KNN Classifier with K value and build
    classifier = Classifier(classname=KNN, options=["-K", K_VALUE])
    classifier.build_classifier(data)

    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, 10, Random(1))


#https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/classifiers/classifiers.py
    accurity = evaluation.percent_correct
    results = evaluation.summary().split('\n')
    print(evaluation.percent_correct, end="\n"+"**"*30+"\n")
    print(evaluation.summary(), end="\n"+"**"*30+"\n")
    print(evaluation.class_details(), end="\n"+"**"*30+"\n")
    print(evaluation.matrix(), end="\n"+"**"*30+"\n")

    '''
    for idx, inst in enumerate(data):
        pred = classifier.classify_instance(inst)
        dist = classifier.distribution_for_instance(inst)
        print(f"{idx + 1}: label index={pred}, class distribution={dist}")
    '''
    

    '''
    # Evaluation
    evaluation = Evaluation(data)
    print(evaluation.summary())
    print("pctCorrect: " + str(evaluation.percent_correct))
    print("incorrect: " + str(evaluation.incorrect))

    # Print classifier
    print(classifier)
    '''

    # Stop JVM
    jvm.stop()

    '''
    df = pd.read_csv(DATASET_FILE, sep=";")

    print(df)

    jvm.start()
    cls = Classifier(classname="weka.classifier.trees.J48")
    print(cls)
    jvm.stop()

    '''


if __name__ == "__main__":
    os.system("cls" if os.name == 'nt' else "clear")

    parser = argparse.ArgumentParser(description="Artificial Intelligence project")

    parser.add_argument("-k", help="K value for K-NN classifier")

    args = parser.parse_args()

    k = args.k if args.k else "5"
    classify(k)