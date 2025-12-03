"""Implementation of the MuLaTabICL Multilabel prediction for the HIV drug resistance dataset"""

# Setup Imports
import pandas as pd
import numpy as np


from sklearn.model_selection import KFold

import data_preprocessing
import prediction_handler
import result_handler

# Baseline Imports
from MultiLabelPFN.src.tabicl import TabICLClassifier



def main():

    # setting datasets
    files = [r"../data/INI_DataSet.txt", r"../data/PI_DataSet.txt", r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt"]


    # setting models
    models = [
        "finetune_step-1250.ckpt",
        "finetune_stage3_step-50.ckpt",
        "bal_prior_step-1000.ckpt",
        "finetune_step-2000.ckpt",
        "bal_prior_step-1350.ckpt",
        "finetune_step-3150.ckpt",
        "full_model_step-1000.ckpt",
        "tabicl-classifier-v1.1-0506.ckpt",
        "random_test/step-1.ckpt",
        "finetune_step-3300.ckpt",
        "bal_prior_step-1700.ckpt",
        "full_model_step-2900.ckpt",
        "label_enc_step-1000.ckpt",
        "label_enc_step-2300.ckpt",
        "tabpfn_full_step-1000.ckpt",
        "tabpfn_full_step-1600.ckpt",
        "zlpr_loss_step-300.ckpt",
        "cat_feat_step-1000.ckpt",
        "cat_feat_step-3350.ckpt",
        "label_enc_step-4300.ckpt",
        "zlpr_loss_step-700.ckpt"
    ]


    version = "_new_loss_sigmoid"


    # kfold splitting
    folds = 5

    kf = KFold(n_splits=folds, random_state=42, shuffle=True)


    for file in files:

        # load dataset
        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)

        for model in models:

            # Setting up the model for prediction
            multi_target_pfn = TabICLClassifier(model_path="../my/" + model, allow_auto_download=False, average_logits=True,
                                                n_jobs=2, verbose=True, use_hierarchical=False)


            # Cross validation
            y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single",
                                                           method="predict_proba")

            df_y_true = pd.DataFrame(y_true, columns=drugs)


            # transformation of probabilities into labels
            y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

            # getting k-fold split
            kfolds = result_handler.get_kfold(kf, X, Y)

            # saving labels
            result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_" + str(folds) + "_fold" + version))

            # saving probabilities
            result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_probabilities_" + str(folds) + "_fold" + version))


if __name__ == '__main__':
    main()