import os
import nbformat
from jinja2 import Template
from nbconvert.preprocessors import ExecutePreprocessor

def linear_regression_generator(features_path, labels_path):
  with open(os.path.join("src","model_templates", "linear_regression.ipynb")) as f:
    template = Template(f.read())

  parsed_template = template.render(features_file_path=features_path, labels_file_path=labels_path)

  with open(os.path.join("out","parsed_linear_regression.ipynb"), "w") as fh:
    fh.write(parsed_template)

  with open(os.path.join("out","parsed_linear_regression.ipynb")) as f:
    nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    print("Executing python notebook...")
    ep.preprocess(nb, {'metadata': {'path': 'out/'}})
    with open(os.path.join('out', 'executed_linear_regression.ipynb'), 'w', encoding='utf-8') as f:
      nbformat.write(nb, f)



  


