import json
import os


class ShapReport:

    def __init__(self):
        self.report_id = None
        self.execution_time = None

        self.original_text = None
        self.original_label = None
        self.original_prediction = None

        self.values=None
        self.data = None
        self.positive = None
        self.negative = None
        self.prediction_without_positive = None
        self.prediction_without_negative = None

    def fit(self, report_id, execution_time,
            original_text, original_label, original_prediction,
            values, data, positive, negative, prediction_without_positive, prediction_without_negative ):
        self.report_id = report_id
        self.execution_time = execution_time

        self.original_text = original_text
        self.original_prediction = original_prediction
        self.original_label = original_label


        self.values = values
        self.data = data
        self.positive = positive
        self.negative = negative
        self.prediction_without_positive = prediction_without_positive
        self.prediction_without_negative = prediction_without_negative



    def save_local_explanation_report(self, output_path, input_name=None):
        """ Save the report to disk. """

        if input_name is not None:
            report_name = "local_explanation_report_{}_{}.json".format(str(input_name), self.report_id)
        else:
            report_name = "local_explanation_report_{}.json".format(self.report_id)

        # Convert the local explanation report class into a dictionary
        explanation_report_dict = self.local_explanation_report_to_dict()

        with open(os.path.join(output_path, report_name), "w") as fp:
            json.dump(explanation_report_dict, fp)
        return

    def local_explanation_report_metadata_to_dict(self):
        metadata = {"report_id": self.report_id,
                    "execution_time": self.execution_time,
                    }
        return metadata

    def local_explanation_report_input_info_to_dict(self):
        input_info = {"original_text": self.original_text,
                      "original_label": self.original_label,
                      "original_prediction": self.original_prediction
                      }
        return input_info

    def local_explanation_report_to_dict(self):
        """ Converts a single local explanation report to dictionary. """
        metadata = self.local_explanation_report_metadata_to_dict()

        input_info = self.local_explanation_report_input_info_to_dict()

        local_explanations_dict = {"values": self.values,
                                   "data": self.data,
                                   "positive": self.positive,
                                   "negative": self.negative,
                                   "prediction_without_positive": self.prediction_without_positive,
                                   "prediction_without_negative": self.prediction_without_negative}

        local_explanation_report_dict = {"metadata": metadata, "input_info": input_info,
                                         "local_explanations": local_explanations_dict}

        return local_explanation_report_dict

    def loadFromJson(self, path):
        with open(path) as explanation_report_json:
            limeReportDict = json.load(explanation_report_json)
            self.report_id = limeReportDict["metadata"]["report_id"]
            self.execution_time = limeReportDict["metadata"]["execution_time"]
            self.num_features = limeReportDict["metadata"]["num_features"]
            self.num_samples = limeReportDict["metadata"]["num_samples"]

            self.original_text = limeReportDict["input_info"]["original_text"]
            self.original_prediction = limeReportDict["input_info"]["original_prediction"]
            self.original_label = limeReportDict["input_info"]["original_label"]

            self.explanation = limeReportDict["local_explanations"]["local_explanations"]
            self.prediction_without_positive = limeReportDict["local_explanations"]["prediction_without_positive"]
            self.prediction_without_negative = limeReportDict["local_explanations"]["prediction_without_negative"]
        return