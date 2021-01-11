import types
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

def classify(k, cross_validation, percentage_split):
    DATASET_FILE    = os.path.join("diabetes_data_upload.csv")
    KNN             = "weka.classifiers.lazy.IBk"
    K_VALUE         = k
    

    # Start JVM
    try:
        jvm.start()
    except RuntimeError:
        jvm.stop()
        jvm.start()

    # Clear screen
    os.system("clear | cls")

    try:
        # Create a Loader object
        loader = Loader(classname="weka.core.converters.CSVLoader")

        # Load the dataset
        data = loader.load_file(DATASET_FILE)

        # Indicates that class is the last attribute
        data.class_is_last()


        # Create KNN Classifier with K value and build
        classifier = Classifier(classname=KNN, options=["-K", str(K_VALUE)])

        results = {}
        
        if cross_validation:
            results["cross_validation"] = {}
            evaluation = Evaluation(data)
            evaluation.crossvalidate_model(classifier, data, cross_validation, Random(1))

            results["cross_validation"]["accurity"]         = evaluation.percent_correct
            results["cross_validation"]["num_instances"]    = evaluation.num_instances
            results["cross_validation"]["class_details"]    = evaluation.class_details()
            results["cross_validation"]["summary"]          = evaluation.summary()
            results["cross_validation"]["matrix"]           = evaluation.matrix()
            

        if percentage_split:
            # evaluate model on train/test split
            results["percentage_split"] = {}

            evaluation = Evaluation(data)
            evaluation.evaluate_train_test_split(classifier, data, percentage_split, Random(1))

            results["percentage_split"]["accurity"]         = evaluation.percent_correct
            results["percentage_split"]["num_instances"]    = evaluation.num_instances
            results["percentage_split"]["class_details"]    = evaluation.class_details()
            results["percentage_split"]["summary"]          = evaluation.summary()
            results["percentage_split"]["matrix"]           = evaluation.matrix()
            
        
        for mode, results in results.items():
            print(f"{mode}")
            for key, result in results.items():
                print(f"\t{key}")
                print(f"\t\t{result}")
                print('\t'+'-'*10)
            print(f"\n\n{'*'*20}\n\n")

        # Stop JVM
        jvm.stop()
    except Exception as e:
        print(e)
        jvm.stop()
    


    

if __name__ == "__main__":
    os.system("cls" if os.name == 'nt' else "clear")

    parser = argparse.ArgumentParser(description="Artificial Intelligence project")

    parser.add_argument("-k", help="K value for K-NN classifier", type=int)
    parser.add_argument("-V", help="Verbose mode")
    parser.add_argument("--cross_validation", help="Evaluate cross validation with given folks count", type=int)
    parser.add_argument("--percentage_split", help="Evaluate percentage split with given test rate", type=float)
    

    args = parser.parse_args()

    if args.V:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    

    k = args.k if args.k is not None else 5
    if k < 1:
        logging.warning("K value can't be smaller than 1")
        raise ValueError("K value can't be smaller than 1")

    if k % 2 == 0:
        logging.warning("K value can't be even numbers")
        raise ValueError("K value can't be even numbers")

    if k > 520:
        logging.warning("K value can't be bigger than instance size")
        raise ValueError("K value can't be bigger than instance size")
    
    logging.info(f"K -> {k}")
    classify(k, args.cross_validation, args.percentage_split)


# https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/classifiers/train_test_split.py
# https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/classifiers/classifiers.py

"""
92.5
************************************************************

Correctly Classified Instances         481               92.5    %
Kappa statistic                          0.8446
Mean absolute error                      0.0748
Root mean squared error                  0.2219
Relative absolute error                 15.7972 %
Root relative squared error             45.619  %
Total Number of Instances              520

************************************************************      
=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class   
                 0,906    0,045    0,970      0,906    0,937      0,848    0,985     0,989     Positive
                 0,955    0,094    0,864      0,955    0,907      0,848    0,985     0,968     Negative
Weighted Avg.    0,925    0,064    0,929      0,925    0,926      0,848    0,985     0,981

************************************************************
=== Confusion Matrix ===

   a   b   <-- classified as
 290  30 |   a = Positive
   9 191 |   b = Negative

************************************************************
"""