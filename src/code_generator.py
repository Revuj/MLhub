import os
import uuid
import nbformat
from jinja2 import Environment, FileSystemLoader
from nbconvert.preprocessors import ExecutePreprocessor


def get_template(template_name, problem_type):
    with open(os.path.join("src", "model_templates", template_name)) as f:
        template = Environment(loader=FileSystemLoader(searchpath=os.path.join("src", "model_templates"))).from_string(
            f"{{% extends '{problem_type}_template.ipynb' %}}\n" + f.read())  # prepend base template to file
    return template


def execute_template(parsed_template, notebook_name, add_unique=True):
    unique_string = ""
    if add_unique:
        unique_string = "_" + str(uuid.uuid4())

    with open(os.path.join("out", f"parsed_{notebook_name}{unique_string}.ipynb"), "w") as fh:
        fh.write(parsed_template)

    with open(os.path.join("out", f"parsed_{notebook_name}{unique_string}.ipynb")) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        print("Executing python notebook...")
        ep.preprocess(nb, {'metadata': {'path': 'out/'}})
        with open(os.path.join('out', f'executed_{notebook_name}{unique_string}.ipynb'), 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


def generate_code(model, train_split, features_path, labels_path, category_threshold):
    model_type = model["type"]
    filename = model["name"] if model["name"] is not None else model_type
    problem_type = model_type.split('_')[-1]
    template = get_template(f"{model_type}.ipynb", problem_type)
    parsed_template = template.render(model=model, train_split=train_split,
                                      features_file_path=features_path, labels_file_path=labels_path, category_threshold=category_threshold)
    execute_template(parsed_template, filename, add_unique=False)
    return f"Executed {filename} model"
