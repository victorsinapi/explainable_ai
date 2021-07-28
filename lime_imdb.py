from lime.lime_text import LimeTextExplainer
import numpy as np
from utils_model.BertModelWrapper import BertModelWrapper
import pandas as pd
from utils.report_lime import LimeReport
import time

#Configuration


#exemple for binary classification on sentiment analysis
model_path = "saved_models/bert_imdb"
dataset_path = "datasets/df_test.csv"
batch_size = 128
class_names = ['Negative', 'Positive']
text_id = "text"
label_id = "label"
number_of_texts = 100
features_percentage = 0.25
num_samples = 5000
output_path = "outputs/lime/lime_imdb"
def getLabel(prediction):
    return 1

'''#exemple for multiclass classification
model_path = "saved_models/bert_agnews"
dataset_path = "datasets/agnews.csv"
batch_size = 128
class_names = ['World', 'Sport','Business','Sci/Tech']
text_id = "Description"
label_id = "Class Index"
number_of_texts = 100
features_percentage = 0.25
num_samples = 5000
output_path = "outputs/lime/lime_agnews"
def getLabel(prediction):
    return int(np.asarray(prediction).argmax())'''



def bert_classifier(texts):
    '''classifier prediction probability function, which takes a list of d strings and outputs a (d, k) numpy array
    with prediction probabilities, where k is the number of classes. For ScikitClassifiers ,
    this is classifier.predict_proba.'''
    # predictions = model_wrapper.predict(texts)
    return model_wrapper.predict(texts)

def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


if __name__ == "__main__":

    model_import_start = time.time()

    # load model
    model_wrapper = BertModelWrapper(model_path, batch_size=batch_size)
    print("Model loaded")
    print(f"Model imported time {time.time() - model_import_start}")

    # load dataset
    df = load_dataset_from_csv(dataset_path)
    print("Dataset loaded")
    print(f"Model imported time {time.time() - model_import_start}")

    # create explainer
    explainer = LimeTextExplainer(class_names=class_names, bow=False, random_state=1)
    print("Explenator created")
    print(f"Model imported time {time.time() - model_import_start}")

    # explanation

    for idx in range(number_of_texts):
        # select a text
        idx = idx
        text = df[text_id][idx]
        original_label = int(df[label_id][idx])
        prediction = model_wrapper.predict([text])[0].tolist()
        label = getLabel(prediction)
        #label = int(np.asarray(prediction).argmax())
        print("pred", prediction)

        explaination_start_time = time.time() - model_import_start
        print(f"Explaination start time {time.time() - model_import_start}")

        num_features = int(len(text.split()) * features_percentage)
        print("Num features : ", num_features)

        # explain
        exp = explainer.explain_instance(text, bert_classifier, num_features=num_features, num_samples=num_samples, labels=[label,])

        explaination_end_time = time.time() - model_import_start
        explaination_time = explaination_end_time - explaination_start_time

        # Creo il testo senza parole evidenziate da Lime
        listExp = [(exp.domain_mapper.indexed_string.word(x[0]),
                    int(exp.domain_mapper.indexed_string.string_position(x[0])),
                    x[1]) for x in exp.local_exp[label]]
        listExp.sort(key=lambda tup: tup[1], reverse=True)
        positive_words = set()
        negative_words = set()
        text_without_positive = text
        text_without_negative = text
        for expl in listExp:
            if (float(expl[2]) > 0):
                positive_words.add(expl[0])
                text_without_positive = text_without_positive[:expl[1]] + text_without_positive[expl[1] + len(expl[0]):]
            else:
                negative_words.add(expl[0])
                text_without_negative = text_without_negative[:expl[1]] + text_without_negative[expl[1] + len(expl[0]):]

        print("Positive words: " + str(positive_words))
        print("Negative words: " + str(negative_words))

        # Nuova predizione
        prediction_without_positive = model_wrapper.predict([text_without_positive])
        prediction_without_negative = model_wrapper.predict([text_without_negative])

        # report

        # Create report
        report = LimeReport()
        report.fit(idx,
                   explaination_time,
                   num_features,
                   num_samples,
                   text,
                   prediction,
                   original_label,
                   listExp,
                   prediction_without_positive[0].tolist(),
                   prediction_without_negative[0].tolist())
        report.save_local_explanation_report(output_path, "25per")

        print("Explaination:")
        print("Text: " + text)
        print("Original label: " + str(original_label))
        print("Prediction: " + str(prediction))
        print("Time: " + str(explaination_time))
        exp.save_to_file(output_path+"/text_25features_lime_bert_on_text" + str(idx) + ".html")