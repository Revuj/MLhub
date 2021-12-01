import os
import nbformat
from jinja2 import Template
from nbconvert.preprocessors import ExecutePreprocessor

def get_template(template_name):
  with open(os.path.join("src","model_templates", template_name)) as f:
      template = Template(f.read())
  return template

def execute_template(parsed_template, notebook_name):
  with open(os.path.join("out",f"parsed_{notebook_name}.ipynb"), "w") as fh:
    fh.write(parsed_template)

  with open(os.path.join("out",f"parsed_{notebook_name}.ipynb")) as f:
    nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    print("Executing python notebook...")
    ep.preprocess(nb, {'metadata': {'path': 'out/'}})
    with open(os.path.join('out', f'executed_{notebook_name}.ipynb'), 'w', encoding='utf-8') as f:
      nbformat.write(nb, f)

def linear_regression_generator(features_path, labels_path):
  template = get_template("linear_regression.ipynb")
  parsed_template = template.render(features_file_path=features_path, labels_file_path=labels_path)
  execute_template(parsed_template, "linear_regression")

def polynomial_regression_generator(features_path, labels_path):
  template = get_template("polynomial_regression.ipynb")
  parsed_template = template.render(features_file_path=features_path, labels_file_path=labels_path)
  execute_template(parsed_template, "polynomial_regression")


  


