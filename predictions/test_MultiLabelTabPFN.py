"""Implementation of the Binary relevance Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold

import data_preprocessing
import prediction_handler
import result_handler

# Baseline Imports


from MultiLabelPFN.src.tabicl import TabICLClassifier
from MultiLabelTabPFN.src.tabpfn import TabPFNClassifier

def main():


    files = [r"../data/Other_datasets/scene.csv", r"../data/Other_datasets/yeast.csv"]
    #models = ["finetune_step-3150.ckpt", r"full_model_step-1000.ckpt", "bal_prior_step-1000.ckpt", "bal_prior_step-1350.ckpt"]
    #files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
    #         r"../data/NNRTI_DataSet.txt"]

    #files = [r"../data/NRTI_DataSet.txt",
    #         r"../data/NNRTI_DataSet.txt"]



    #models = ["tabpfn_full_step-4000.ckpt",
    #    "tabpfn_full_step-5000.ckpt",
    #          "tabpfn_full_step-5300.ckpt"]

    #models = ["label_enc_step-1000.ckpt",
    #    "label_enc_step-2300.ckpt"]
    models = ["tabpfn_full_step-1000.ckpt", "tabpfn_full_step-1600.ckpt","tabpfn_full_step-3000.ckpt", "tabpfn_full_step-4000.ckpt",
        "tabpfn_full_step-5000.ckpt",
              "tabpfn_full_step-5300.ckpt"]

    feature_prefix = "F"
    label_prefix = "T"



    use_kfold = True
    folds = 5



    for file in files:

        version = "_" + file.split("/")[-1].strip(".csv") + "_dataset"

        #X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)


        df = pd.read_csv(file, true_values=["b'1'"], false_values=["b'0'"], dtype=float)

        X = df.filter(regex=feature_prefix)
        Y = df.filter(regex=label_prefix)
        
        #print(X)
        #print(Y)

        drugs= list(Y.columns.values)


        for model in models:
            #multi_target_pfn = TabICLClassifier(model_path="../my/random_test/step-1.ckpt", allow_auto_download=False, n_jobs=2, verbose=True, use_hierarchical=False)

            #multi_target_pfn = TabICLClassifier(model_path="../my/" + model, allow_auto_download=False, average_logits=True,
            #                                    n_jobs=2, verbose=True, use_hierarchical=False)

            multi_target_pfn = TabPFNClassifier(model_path="../my/" + model, n_preprocessing_jobs=1, random_state=42, average_before_softmax=True)


            if use_kfold == False:

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

                #print("y_test: ", y_test.shape)

                trained_model_pfn = multi_target_pfn.fit(X_train, y_train)

                y_pred = trained_model_pfn.predict(X_test)


                y_pred_df = pd.DataFrame(y_pred, columns=drugs)

                y_test_df = pd.DataFrame(y_test, columns=drugs)




                result_handler.save_multilabel(y_pred_df, y_test_df, label= (file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[0] + "_MuLaTabICL_random_step1"))


                y_pred_proba = trained_model_pfn.predict_proba(X_test)

                y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=drugs)

                result_handler.save_multilabel(y_pred_proba_df, y_test_df, label=(
                            file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                        0] + "_MuLaTabICL_probabilities_random_step1"))

            else:

                kf = KFold(n_splits=folds, random_state=42, shuffle=True)

                print("Before CV: ", X.shape)

                y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single", method="predict_proba")

                #print(y_pred, y_true)

                df_y_true = pd.DataFrame(y_true, columns=drugs)

                y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

                y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

                kfolds = result_handler.get_kfold(kf, X, Y)


                result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                            file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                        0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_"+ str(folds) + "_fold" + version))



                result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_MuLaTabICL_" + model.strip(".ckpt") + "_probabilities_"+ str(folds) + "_fold" + version))



if __name__ == '__main__':
    main()