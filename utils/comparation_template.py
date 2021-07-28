comparationTemplate = '''
<html>
<head>
  <meta content="text/html; charset=ISO-8859-1"
 http-equiv="content-type">
  <title>Comparation Report</title>
</head>
<script>
function showHideExplaination(id) {{
 var x = document.getElementById(id);
  if (x.style.display === "none") {{
    x.style.display = "block";
  }} else {{
    x.style.display = "none";
  }}}}
</script>
<body>

    <!-- Tab content -->
    <div id="input" class="tabcontent">
      <h1>INPUT INFO</h1>
      <h3>Original Text</h3>
      <div class="boxed">{original_text}</div>

      <div id="input_info">
      <table id="table_input">
          <tr>
            <th>Original Probabilities</th>
            <th>Original Label</th>
          </tr>
          <tr>
            <th>{original_probabilities}</th>
            <th>{original_label}</th>
          </tr>
      </table>
    </div>
    </div>

<table style="border-collapse: collapse; width: 100%; height: 205px;" border="1">
<tbody>
<tr style="height: 34px;">
<td style="width: 1%; height: 34px;">&nbsp;</td>
<td style="width: 19%; height: 34px;">Lime</td>
<td style="width: 20%; height: 34px;">T-Ebano (MLWE)</td>
<td style="width: 20%; height: 34px;">T-Ebano (POS)</td>
<td style="width: 20%; height: 34px;">T-Ebano (SEN)</td>
<td style="width: 20%; height: 34px;">SHAP</td>
</tr>
<tr style="height: 62px;">
<td style="width: 1%; height: 62px;"><p>Highlighted Text</p></td>
<td style="width: 19%; height: 62px;">{Lime_highlighted_text}</td>
<td style="width: 20%; height: 62px;">{MLWE_hightlighted_text}</td>
<td style="width: 20%; height: 62px;">{POS_hightlighted_text}</td>
<td style="width: 20%; height: 62px;">{SEN_hightlighted_text}</td>
<td style="width: 20%; height: 62px;">{Shap_highlighted_text}</td>
</tr>
<tr style="height: 45px;">
<td style="width: 1%; height: 45px;"><p>Explanations</p></td>
<td style="width: 19%; height: 45px;">{Lime_explanations}</td>
<td style="width: 20%; height: 45px;">{MLWE_explanations}</td>
<td style="width: 20%; height: 45px;">{POS_explanations}</td>
<td style="width: 20%; height: 45px;">{SEN_explanations}</td>
<td style="width: 20%; height: 45px;">{Shap_explanations}</td>
</tr>
<tr style="height: 34px;">
<td style="width: 1%; height: 34px;">Perturbed Probabilities</td>
<td style="width: 19%; height: 34px;">{Lime_perturbed}</td>
<td style="width: 20%; height: 34px;">{MLWE_perturbed}</td>
<td style="width: 20%; height: 34px;">{POS_perturbed}</td>
<td style="width: 20%; height: 34px;">{SEN_perturbed}</td>
<td style="width: 20%; height: 34px;">{Shap_perturbed}</td>

</tr>
<tr style="height: 30px;">
<td style="width: 1%; height: 30px;">Time</td>
<td style="width: 19%; height: 30px;">{Lime_time}</td>
<td style="width: 20%; height: 30px;">{MLWE_time}</td>
<td style="width: 20%; height: 30px;">{POS_time}</td>
<td style="width: 20%; height: 30px;">{SEN_time}</td>
<td style="width: 20%; height: 30px;">{Shap_time}</td>
</tr>
</tbody>
</table>
</body>
</html>'''
