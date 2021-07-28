import shap
import numpy as np
import scipy as sp
from utils_model.BertModelWrapper import BertModelWrapper
import pandas as pd
import time
from utils.report_shap import ShapReport

#Configuration

'''#exemple for binary classification on sentiment analysis
alghoritm = "permutation"
model_path = "saved_models/bert_imdb"
dataset_path = "datasets/df_test.csv"
batch_size = 128
number_of_texts = 100
text_id = "text"
label_id = "label"
def getLabel(prediction):
    return 1
percentage_of_tokens = 0.10
output_path = "outputs/shap/shap_imdb"'''

#exemple for multiclass classification
alghoritm = "permutation"
model_path = "saved_models/bert_agnews"
dataset_path = "datasets/agnews.csv"
batch_size = 128
number_of_texts = 100
text_id = "Description"
label_id = "Class Index"
def getLabel(prediction):
    return int(np.asarray(prediction).argmax())
percentage_of_tokens = 0.10
output_path = "outputs/shap/shap_agnews"



def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

def f(x):
    print(len(x))
    scores = model_wrapper.predict(x.tolist())
    val = sp.special.logit(scores[:,label]) # use one vs rest logit units
    return val


if __name__ == "__main__":

    # load BERT model
    model_wrapper = BertModelWrapper(model_path,batch_size=batch_size)
    df = load_dataset_from_csv(dataset_path)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer


    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer, algorithm=alghoritm)

    for idx in range(number_of_texts):

        idx = idx

        # input info
        original_text = df[text_id][idx]
        original_label = int(df[label_id][idx])
        original_prediction = model_wrapper.predict([original_text])[0].tolist()
        label = getLabel(original_prediction)

        # generate explanation
        explaination_start_time = time.time()
        shap_values = explainer([original_text])
        explaination_end_time = time.time()
        explaination_time = explaination_end_time - explaination_start_time

        # metadata
        report_id = idx
        execution_time = explaination_time

        # local_explaination
        sv = shap_values[0]
        values = sv.values
        base_values = sv.base_values
        data = sv.data

        top = int(len(values) * percentage_of_tokens)
        max = sorted(values)[len(values) - top]
        text_without_positive = []
        positive = []
        for i in range(values.__len__()):
            if values[i] < max:
                text_without_positive.append(data[i])
            else:
                positive.append(data[i])
        text_without_positive = ''.join(text_without_positive)
        positive = ''.join(positive)

        min = sorted(values)[top]
        text_without_negative = []
        negative = []
        for i in range(values.__len__()):
            if values[i] > min:
                text_without_negative.append(data[i])
            else:
                negative.append(data[i])
        text_without_negative = ''.join(text_without_negative)
        negative = ''.join(negative)
        if max < 0:
            text_without_positive = original_text
            positive = None
        if min > 0:
            text_without_negative = original_text
            negative = None

        prediction_without_positive = model_wrapper.predict([text_without_positive])
        prediction_without_negative = model_wrapper.predict([text_without_negative])

        # Create report
        report = ShapReport()
        report.fit(report_id,
                   execution_time,

                   original_text,
                   int(original_label),
                   original_prediction[0].tolist(),

                   values.tolist(),
                   data.tolist(),
                   positive,
                   negative,
                   prediction_without_positive[0].tolist(),
                   prediction_without_negative[0].tolist())

        report.save_local_explanation_report(output_path)