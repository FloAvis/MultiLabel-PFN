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

def main():

    models = [
    "finetune_step-1250.ckpt",
    "bal_prior_step-1000.ckpt",
    "finetune_step-2000.ckpt",
    "bal_prior_step-1350.ckpt",
    "finetune_step-3150.ckpt",
    "full_model_step-1000.ckpt",
    "tabicl-classifier-v1.1-0506.ckpt",
    "random_test/step-1.ckpt",
    "full_model_step-1000.ckpt"]



    for model in models:

        multi_target_pfn = TabICLClassifier(model_path="./my/" + model, allow_auto_download=False, average_logits=True,
                                            n_jobs=2, use_hierarchical=False)

        config_dict = {
            "batch_size": 24,
            "batch_size_per_gp": 4,
            "min_features": 2,
            "max_features": 100,
            "max_labels": 10,
            "min_quan": 0.5,
            "max_quan": 0.95,
            # "min_seq_len": config.min_seq_len,
            "max_seq_len": 1024,
            # "log_seq_len": config.log_seq_len,
            # "seq_len_per_gp": config.seq_len_per_gp,
            # "min_train_size": config.min_train_size,
            # "max_train_size": config.max_train_size,
            # "replay_small": config.replay_small,
            "prior_type": "mix_scm",
            # "device": config.prior_device,
        }

        print("Threshold for model " + model.strip(".ckpt") + ": " + prediction_handler.calc_threshold(multi_target_pfn, config_dict))





if __name__ == '__main__':
    main()