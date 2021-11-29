import os
from jinja2 import Template

def linear_regression_generator(features_path, labels_path):
  with open(os.path.join("src","model_templates", "linear_regression.py")) as f:
    template = Template(f.read())
  print(template.render(features_file_path=features_path, labels_file_path=labels_path))

