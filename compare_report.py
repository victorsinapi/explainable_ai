import csv
from utils import comparator_imdb
import io

out_csv = "outputs/csv/comparation100imdb.csv"

if __name__ == "__main__":

    codice = "00"
    array_csv = []
    indice = int(0)

    array_csv.append(
        [" ", "ID","Text", "Label", "Pred", "Diff", "LIME highlighted percentage", "MLWE highlighted percentage", "POS highlighted percentage", "SEN highlighted percentage","Shap highlighted percentage",
         "LIME pred var", "MLWE red var", "POS pred var", "SEN pred var","Shap pred var",
         "LIME pred var rel", "MLWE  pred var rel", "POS  pred var rel", "SEN  pred var rel","Shap pred var rel",
         "LIME time", "EBANO time","Shap time",
         "LIME class", "MLWE class", "POS class", "SEN class","Shap Class"])

    for i in range (100):

        codice=str(i)

        lime_local_explanation_path ="expLime/25per/local_explanation_report_10per_"+str(i)+".json"
        ebano_local_explanation_path = "expEbano/imdb/local_explanation_report_"+str(i)+".json"
        shap_local_explanation_path = "expShap/permutazioni/local_explanation_report_"+str(i)+".json"

        comparator = comparator.Comparator(code=codice,array_csv=array_csv,index=indice)

        comparator.fit(comparator.LIME, lime_local_explanation_path)

        comparator.addExplanation(comparator.LIME, lime_local_explanation_path )
        comparator.addExplanation(comparator.EBANO,ebano_local_explanation_path)
        comparator.addExplanation(comparator.SHAP,shap_local_explanation_path)

        comparator.save_as_html()
        indice = indice +1


    with io.open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(array_csv)