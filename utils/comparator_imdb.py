import json
import os
import sys
from utils import comparation_template
from utils import report_lime
from utils import report_shap

sys.path.append('../')
from TextEBAnoExpress.explainer import LocalExplanationReport

LIME = "Lime"
EBANO = "Ebano"
SHAP ="Shap"

outputs_path = "outputs/comparations"
fileName = "comparation_"

def html_str_to_file(text, filename):
    """Write a file with the given name and the given text."""
    output = open(os.path.join(outputs_path,filename), "w", encoding="utf-8")
    output.write(text)
    output.close()
    return

def browseLocal(webpageText, filename="tmp.html"):
    """ Start your webbrowser on a local file containing the text with given filename. """
    import webbrowser, os.path
    html_str_to_file(webpageText, filename)
    webbrowser.open("file:///" + os.path.abspath(os.path.join(outputs_path,filename)))
    return

class Comparator:
    def __init__(self,code=0, array_csv=[], index=0):
        self.code=code
        self.array_csv=array_csv
        self.index = index

        return

    def fit(self, explainer, path):
        #Insert the value related to raw text
        if (explainer!=LIME):
            exit("Incorrect explainer value")

        with open(path) as explanation_report_json:
            explanation_report_dict = json.load(explanation_report_json)
            self.original_text = explanation_report_dict["input_info"]["original_text"]
            self.original_label = explanation_report_dict["input_info"]["original_label"]
            self.original_prediction = explanation_report_dict["input_info"]["original_prediction"]

        return

    def addExplanation(self, explainer, path):
        if (explainer==LIME):
            self.limeReport = report_lime.LimeReport()
            self.limeReport.loadFromJson(path)
        elif(explainer==EBANO):
            self.ebanoReport = LocalExplanationReport()
            print("#####",path)
            self.ebanoReport.fit_local_explanation_report_from_json_file(path)
        elif (explainer == SHAP):
            self.shapReport = report_shap.ShapReport()
            self.shapReport.loadFromJson(path)
            return

    #EBANO

    def ebano_perturbed_probabilities(self, local_explanation):
        change_probabilities = [
            local_explanation.perturbed_probabilities[i] - self.ebanoReport.original_probabilities[i]
            for i in range(len(self.ebanoReport.original_probabilities))]

        change_probabilities_string = "[ "
        for p in change_probabilities:
            if (p >= 0):
                change_probabilities_string = change_probabilities_string + """<span id="positive_influential_color">+""" + str(
                    round(p, 3))
            else:
                change_probabilities_string = change_probabilities_string + """<span id="negative_influential_color">""" + str(
                    round(p, 3))
            change_probabilities_string = change_probabilities_string + """</span>  ,  """

        change_probabilities_string = change_probabilities_string + " ]"

        string = "<p>{} - {}</p>".format(
                                    [round(p_p, 3) for p_p in local_explanation.perturbed_probabilities],
                                    change_probabilities_string)

        original = self.original_prediction[1]
        new = local_explanation.perturbed_probabilities[1]

        if original < 0.5 :
            original = 1- original
            new = 1 - new

        string = string + "<p> {} > {} :  {}".format( str(round(original,3)),
                                                      round(new,3),
                                                      round(new-original,3)
                                                    )
        perturbed_val = new - original
        return string, perturbed_val,  perturbed_val/original

    def ebano_highlight_feature_into_text(self, input_positions_tokens, local_explanation, r=0, g=255, b=255, a=1, type="MLWE"):

        id = type+str(local_explanation.perturbation.feature.feature_id)

        html_string = '<p onclick=\'showHideExplaination("'+id+'")\'>'


        html_string = html_string + """Feature {} - nPIR {} - cover ratio {}/{}</p>""".format(
            local_explanation.perturbation.feature.feature_id,
            round(local_explanation.numerical_explanation.nPIR_original_top_class, 3),
            len(local_explanation.perturbation.feature.positions_tokens),
            self.wordCount
        )

        html_string = html_string + '<p id="'+id+'" style="display: none;">'
        feature_color = "background-color:rgba({}, {}, {}, {})".format(r, g, b, a)
        for position in range(len(input_positions_tokens)):
            if str(position) in local_explanation.perturbation.feature.positions_tokens:
                html_string = html_string + '<span style="{}">{}</span> '.format(feature_color,
                                                                                        input_positions_tokens[
                                                                                            position])
            else:
                html_string = html_string + '<span>{}</span> '.format(input_positions_tokens[position])
        html_string = html_string + "</p>"


        return html_string

    def get_html_string_summary_feature_type(self, feature_type=any(["MLWE", "POS", "SEN"])):
        # get features of `feature_type` without combinations
        filtered_local_explanations = self.ebanoReport.get_filtered_local_explanations(feature_type, [1])

        positions_tokens_score = {}  # Dictionary with `position` as key and as value the tuple `(token, nPIR)`

        for le in filtered_local_explanations:
            for position, token in le.perturbation.feature.positions_tokens.items():
                positions_tokens_score[position] = (token, round(le.numerical_explanation.nPIR_original_top_class, 4))

        html_string = ""
        for position in sorted(positions_tokens_score.keys(), key=lambda k: int(k)):
            token_score = positions_tokens_score[position]
            if token_score[1] >= 0:
                positive_color = "background-color:rgba(124, 252, 0, {})".format(token_score[1])
                html_string = html_string + '<span style="{}">{}</span> '.format(positive_color, token_score[0])
            else:
                negative_color = "background-color:rgba(255, 99, 71, {})".format(token_score[1])
                html_string = html_string + '<span style="{}">{}</span> '.format(negative_color, token_score[0])

        return html_string

#LIME
    def lime_get_highlighted(self):
        html_str = "<p>"
        i = 0
        j = 0
        text = self.limeReport.original_text
        explainations = self.limeReport.explanation
        explainations.sort(key=lambda x: x[1])
        max_val = max([exp[2] for exp in explainations])
        min_val = min([exp[2] for exp in explainations])
        while(i < len(text)):
            if j<self.limeReport.num_features and  (i == int(explainations[j][1])):
                if explainations[j][2]>0 :
                    color = "background-color:rgba(124, 252, 0, {})".format(str(explainations[j][2]/max_val))
                else:
                    color = "background-color:rgba(255, 99, 71, {})".format(str(explainations[j][2]/min_val))
                html_str = html_str +'<span style="{}">{}</span>'.format( color, explainations[j][0])
                i = i + len(explainations[j][0])
                j = j + 1
            else:
                html_str = html_str + '{}'.format( text[i])
                i = i +1

        html_str = html_str +"</p>"
        return html_str

    def lime_get_exp_html(self):
        html_str = ""
        explainations = self.limeReport.explanation

        max_val = max([exp[2] for exp in explainations])
        min_val = min([exp[2] for exp in explainations])

        explainations.sort(key=lambda x: x[2])

        highlighted_count = 0
        original = self.original_prediction[1]

        for exp in explainations:
            if exp[2] > 0:
                color = "background-color:rgba(124, 252, 0, {})".format(str(exp[2] / max_val))
                html_str = html_str + '<p>{}    <span  style="{}">{}</span></p>'.format(exp[0],color,str(exp[2]))
                if original > 0.5:
                    highlighted_count += 1
            else:
                color = "background-color:rgba(255, 99, 71, {})".format(str(exp[2] / min_val))
                html_str =  '<p>{}    <span  style="{}">{}</span></p>'.format(exp[0], color, exp[2]) + html_str
                if original < 0.5:
                    highlighted_count += 1
        html_str = "<p>"+html_str +"</p>"
        return html_str, highlighted_count

    def lime_get_perturbed_html (self):
        original = self.original_prediction[1]
        if ( original >0.5 ):
            new = self.limeReport.prediction_without_positive[1]
        else:
            new = self.limeReport.prediction_without_negative[1]
            new = 1 - new
            original = 1 - original

        html_str = '<p>{} -> {}  :    {}  </p>'.format(str(round(original, 3)),
                                                       str(round(new, 3)),
                                                       str(round(-original + new, 3)))
        variation = -original + new
        return html_str, variation, variation/original

#SHAP

    def shapGetSummary(self):
        html_str = "<p>"
        tokens = self.shapReport.data
        values = self.shapReport.values
        max_val = max(values)
        min_val = min(values)

        for i in range(len(values)):
            if values[i] >0:
                color = "background-color:rgba(124, 252, 0, {})".format(str(values[i]/max_val))
            else:
                if min_val == 0:
                    opacity = 0
                else:
                    opacity = values[i]/ min_val
                color = "background-color:rgba(255, 99, 71, {})".format(str(opacity))
            html_str = html_str + '<span style="{}">{}</span>'.format(color, tokens[i])

        html_str = html_str +"</p>"
        return html_str

    def shapGetExplainations(self):
        html_str = "<p>"
        pred = self.shapReport.original_prediction[1]

        if pred > 0.5 :
            expl = self.shapReport.positive
        else:
            expl = self.shapReport.negative

        html_str = html_str + str(expl) + "</p>"
        highlighted_count = len(str(expl).lower().split())

        return html_str, highlighted_count

    def shapGetPerturbed(self):
        original = self.original_prediction[1]
        if (original > 0.5):
            new = self.shapReport.prediction_without_positive[1]
        else:
            new = self.shapReport.prediction_without_negative[1]
            new = 1 - new
            original = 1 - original

        html_str = '<p>{} -> {}  :    {}  </p>'.format(str(round(original, 3)),
                                                       str(round(new, 3)),
                                                       str(round(-original + new, 3)))
        variation = -original + new
        return html_str, variation, variation/original

    def save_as_html(self):
        self.wordCount = max(len(self.original_text.split(" ")), len(self.ebanoReport.positions_tokens))


        '''Lime elaboration'''
        lime_summary = self.lime_get_highlighted()
        lime_explaination, lime_count_highlighted = self.lime_get_exp_html()
        lime_perturbed, lime_perturbed_val, lime_perturbed_rel= self.lime_get_perturbed_html()

        '''Shap Elaboration'''
        shap_summary = self.shapGetSummary()
        shap_explaination, shap_count_highlighted = self.shapGetExplainations()
        shap_perturbed, shap_perturbed_val, shap_perturbed_rel= self.shapGetPerturbed()

        '''Ebano elaboration'''
        html_mlwe_summary = self.get_html_string_summary_feature_type("MLWE")
        html_pos_summary = self.get_html_string_summary_feature_type("POS")
        html_sen_summary = self.get_html_string_summary_feature_type("SEN")

        MLWE_explainations = ""
        count = 0


        def eval_score1 (percentage, perturbed_variation):
            if perturbed_variation <= 0 or percentage == 1:
                return 0

            percentage = 1-percentage
            res = 1/percentage + 1/perturbed_variation
            if res==0:
                return 0
            return 2/res

        def percentage (local_explanation):
            return len(local_explanation.perturbation.feature.positions_tokens)/self.wordCount

        MLWE_full = self.ebanoReport.get_filtered_local_explanations(feature_type_list="MLWE", combination_list=[1, 2])
        localExplanationsMLWE = sorted(MLWE_full,
                                        key=lambda local_explanation: eval_score1(percentage(local_explanation),
                                                                  local_explanation.numerical_explanation.nPIR_original_top_class),
                                       reverse=True)


        for l_e in localExplanationsMLWE:
                current_exp = self.ebano_highlight_feature_into_text(self.ebanoReport.positions_tokens, l_e, type="MLWE")
                MLWE_explainations = MLWE_explainations + current_exp
                count = count+1
                if (count >= 10):
                    MLWE_explainations = MLWE_explainations + ""
                    break

        POS_explainations = ""
        count = 0
        localExplanationsPOS = sorted(self.ebanoReport.get_filtered_local_explanations(feature_type_list="POS", combination_list=[1, 2]),
                                      key=lambda local_explanation: eval_score1(percentage(local_explanation),
                                                                               local_explanation.numerical_explanation.nPIR_original_top_class),
                                      reverse=True)


        for l_e in localExplanationsPOS:
                current_exp = self.ebano_highlight_feature_into_text(self.ebanoReport.positions_tokens, l_e, type="POS")
                POS_explainations = POS_explainations + current_exp
                count = count+1
                if (count >= 10):
                    POS_explainations = POS_explainations + ""
                    break

        SEN_explainations = ""
        count = 0
        localExplanationsSEN = sorted(self.ebanoReport.get_filtered_local_explanations(feature_type_list="SEN", combination_list=[1, 2]),
                                      key=lambda local_explanation: eval_score1(percentage(local_explanation),
                                                                               local_explanation.numerical_explanation.nPIR_original_top_class),
                                      reverse=True)
        for l_e in localExplanationsSEN:
                current_exp = self.ebano_highlight_feature_into_text(self.ebanoReport.positions_tokens, l_e, type="SEN")
                SEN_explainations = SEN_explainations + current_exp
                count = count+1
                if (count >= 10):
                    SEN_explainations = SEN_explainations + ""
                    break

        MLWE_Perturbed, MLWE_perturbed_val,MLWE_perturbed_rel = self.ebano_perturbed_probabilities(localExplanationsMLWE.__getitem__(0))
        POS_Perturbed, POS_perturbed_val, POS_perturbed_rel = self.ebano_perturbed_probabilities(localExplanationsPOS.__getitem__(0))
        SEN_Perturbed, SEN_perturbed_val,SEN_perturbed_rel = self.ebano_perturbed_probabilities(localExplanationsSEN.__getitem__(0))

        original = '<p>lime [ {}  ,   {} ]<br>ebano [ {}  ,   {} ]</p>'.format(str(round(1-self.original_prediction[1],5)),
                                                                        str(round(self.original_prediction[1],5)),
                                                                               str(round(self.ebanoReport.original_probabilities[0],5)),
                                                                               str(round(self.ebanoReport.original_probabilities[1], 5))
                                                                               )
        contents = comparation_template.comparationTemplate.format(
                            original_text = self.original_text,
                            original_probabilities = original,
                            original_label = self.original_label,

                            Lime_highlighted_text = lime_summary,
                            MLWE_hightlighted_text =html_mlwe_summary,
                            POS_hightlighted_text = html_pos_summary,
                            SEN_hightlighted_text = html_sen_summary,
                            Shap_highlighted_text = shap_summary,

                            Lime_explanations = lime_explaination,
                            MLWE_explanations = MLWE_explainations,
                            POS_explanations = POS_explainations,
                            SEN_explanations = SEN_explainations,
                            Shap_explanations = shap_explaination,

                            Lime_perturbed = lime_perturbed,
                            MLWE_perturbed = MLWE_Perturbed,
                            POS_perturbed= POS_Perturbed,
                            SEN_perturbed= SEN_Perturbed,
                            Shap_perturbed = shap_perturbed,

                            Lime_time= str(round(self.limeReport.execution_time)) + " sec",
                            MLWE_time= str(round(self.ebanoReport.execution_time)) + " sec",
                            POS_time= str(round(self.ebanoReport.execution_time)) + " sec",
                            SEN_time= str(round(self.ebanoReport.execution_time)) + " sec",
                            Shap_time = str(round(self.shapReport.execution_time)) + " sec"
                    )
        #generiamo il csv

        #highligthed ratio calculation
        lime_highlighted_ratio = lime_count_highlighted/self.wordCount
        shap_highlighted_ratio = shap_count_highlighted/self.wordCount

        mlwe_exp = localExplanationsMLWE.__getitem__(0)
        pos_exp = localExplanationsPOS.__getitem__(0)
        sen_exp = localExplanationsSEN.__getitem__(0)
        MLWE_highlighted_ratio = len(mlwe_exp.perturbation.feature.positions_tokens)/self.wordCount
        POS_highlighted_ratio = len(pos_exp.perturbation.feature.positions_tokens)/self.wordCount
        SEN_highlighted_ratio = len(sen_exp.perturbation.feature.positions_tokens)/self.wordCount

        self.array_csv.append([self.index, self.code, self.original_text, self.original_label,
                        self.original_prediction[1], self.shapReport.original_prediction[1],

                         lime_highlighted_ratio, MLWE_highlighted_ratio, POS_highlighted_ratio, SEN_highlighted_ratio, shap_highlighted_ratio,

                          lime_perturbed_val,  MLWE_perturbed_val, POS_perturbed_val, SEN_perturbed_val, shap_perturbed_val,

                          lime_perturbed_rel, MLWE_perturbed_rel, POS_perturbed_rel, SEN_perturbed_rel, shap_perturbed_rel,

                          self.limeReport.execution_time, self.ebanoReport.execution_time, self.shapReport.execution_time,

                          eval_score1(lime_highlighted_ratio, -lime_perturbed_rel) , eval_score1(MLWE_highlighted_ratio, -MLWE_perturbed_rel),
                          eval_score1(POS_highlighted_ratio, -POS_perturbed_rel), eval_score1(SEN_highlighted_ratio, -SEN_perturbed_rel),
                          eval_score1(shap_highlighted_ratio, -shap_perturbed_rel)

                          ])

        fname = fileName+self.code+".html"
        html_str_to_file(contents, fname)
        #browseLocal(contents, filename=fname)
        return