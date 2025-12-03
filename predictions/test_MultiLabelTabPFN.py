"""Implementation of the MuLaTabPFN Multilabel prediction for the HIV drug resistance dataset"""

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

import data_preprocessing
import prediction_handler
import result_handler

# Baseline Imports

from MultiLabelTabPFN.src.tabpfn import TabPFNClassifier

def main():

    # setting datasets
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt"]

    # setting the models
    models = ["tabpfn_full_step-1000.ckpt", "tabpfn_full_step-1600.ckpt","tabpfn_full_step-3000.ckpt", "tabpfn_full_step-4000.ckpt",
        "tabpfn_full_step-5000.ckpt",
              "tabpfn_full_step-5300.ckpt"]

    # kfold splitting
    folds = 5

    kf = KFold(n_splits=folds, random_state=42, shuffle=True)

    for file in files:

        # naming scheme for data
        version = "_" + file.split("/")[-1].strip(".csv") + "_dataset"

        #load dataset
        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)


        for model in models:

            # Setting up the model for prediction
            multi_target_pfn = TabPFNClassifier(model_path="../my/" + model, n_preprocessing_jobs=1, random_state=42, average_before_softmax=True)



            # Cross validation
            y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single", method="predict_proba")

            df_y_true = pd.DataFrame(y_true, columns=drugs)


            # transformation of probabilities into labels
            y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

            # getting k-fold split
            kfolds = result_handler.get_kfold(kf, X, Y)

            # saving labels
            result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_"+ str(folds) + "_fold" + version))


            #saving probabilities
            result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_probabilities_"+ str(folds) + "_fold" + version))



if __name__ == '__main__':
    main()