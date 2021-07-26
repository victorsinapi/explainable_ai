from utils_model.BertModelWrapper import BertModelWrapper
import sys
sys.path.append('../')
from TextEBAnoExpress import explainer
import re
import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from official.nlp.bert import tokenization
import time

#configuration

model_path = "saved_models/bert_imdb"
dataset_path = "datasets/df_test.csv"
batch_size = 128
start_index=0
end_index=99
text_id = "text"
label_id = "label"

'''model_path = "saved_models/bert_agnews"
dataset_path = "datasets/agnews.csv"
batch_size = 128
start_index=0
end_index=99
text_id = "Description"
label_id = "Class Index"'''


def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


if __name__ == "__main__":

    model_import_start = time.time()

    model_wrapper = BertModelWrapper(model_path,batch_size=batch_size)
    df = load_dataset_from_csv(dataset_path)

    print(f"Dataset imported time {time.time() - model_import_start}")

    texts = df[text_id][start_index:end_index+1].tolist()
    true_labels = df[label_id][start_index:end_index+1].tolist()

    print(f"Number texts {len(texts)}")

    exp = explainer.LocalExplainer(model_wrapper, "20201117_bert_model_imdb_reviews_exp_0")

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1] * len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_combinations=True)

    print(f"Model Explaination time {time.time() - model_import_start}")
