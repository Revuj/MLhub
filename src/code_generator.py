import os
import uuid
import nbformat
import shutil
from jinja2 import Environment, FileSystemLoader, Template
from nbconvert.preprocessors import ExecutePreprocessor


def get_template(template_name, problem_type):
    with open(os.path.join("src", "model_templates", template_name)) as f:
        template = Environment(loader=FileSystemLoader(searchpath=os.path.join("src", "model_templates", problem_type))).from_string(
            f"{{% extends 'template.ipynb' %}}\n" + f.read())  # prepend base template to file
    return template


def execute_template(parsed_template, notebook_name, out_path):
    with open(os.path.join(out_path, f"parsed_{notebook_name}.ipynb"), "w") as fh:
        fh.write(parsed_template)

    with open(os.path.join(out_path, f"parsed_{notebook_name}.ipynb")) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        print("Executing python notebook...")
        ep.preprocess(nb, {'metadata': {'path': out_path}})
        with open(os.path.join(out_path, f'executed_{notebook_name}.ipynb'), 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


def generate_statistical_comparison(problem_type, out_path):
    src_path = os.path.join("src", "model_templates", problem_type, "statistical_comparison.ipynb")
    dest_path = os.path.join(out_path, "statistical_comparison.ipynb")
    shutil.copy(src_path, dest_path)

    with open(dest_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        print("Executing statistical comparison python notebook...")
        ep.preprocess(nb, {'metadata': {'path': out_path}})
        with open(dest_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    print("Executed statistical comparison")


def generate_code(model, train, features_path, labels_path, out_path):
    model_type = model["type"]
    model_name = model["name"] if model["name"] is not None else f'{model_type}_{uuid.uuid1()}'
    problem_type = model_type.split('_')[-1]
    template = get_template(os.path.join(problem_type, f"{model_type}.ipynb"), problem_type)
    parsed_template = template.render(model=model, train_split=train["split"],
                                      features_file_path=features_path, labels_file_path=labels_path, 
                                      model_name=model_name, category_threshold=train["data"]["category_threshold"])
    
    execute_template(parsed_template, model_name, out_path)
    return f"Executed {model_name} model"


def generate_cnn(model, train, features_path, labels_path, out_path):
    model_type = model["type"]
    model_name = model["name"] if model["name"] is not None else f'{model_type}_{uuid.uuid1()}'
    classification_input = model["input"]["type"] + "_classification"
    template = get_template(os.path.join(model_type, f"{classification_input}.ipynb"), model_type)
    parsed_template = template.render(model=model, train_split=train["split"],
                                      features_file_path=features_path, labels_file_path=labels_path, 
                                      model_name=model_name, category_threshold=train["data"]["category_threshold"], 
                                      train_data=train["data"], train_settings=train["train_settings"])
    
    execute_template(parsed_template, model_name, out_path)
    return f"Executed {model_name} model"
