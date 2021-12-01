import os
from jinja2 import Template

def linear_regression_generator(features_path, labels_path):
  with open(os.path.join("src","model_templates", "linear_regression.ipynb")) as f:
    template = Template(f.read())

  parsed_template = template.render(features_file_path=features_path, labels_file_path=labels_path)

  with open(os.path.join("out","parsed_linear_regression.ipynb"), "w") as fh:
    fh.write(parsed_template)


