import json
import os
from datetime import datetime
import concurrent.futures
from jsonschema import Draft7Validator, validators
import code_generator


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


def get_schema():

    p = os.path.join("src", "mlhub.json")
    with open(p, 'r') as file:
        schema = json.load(file)
    return schema


DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)


def verify_exists(path):
    if not os.path.exists(path):
        raise Exception(f"File {path} was not found")


def validate_json(json):
    DefaultValidatingDraft7Validator(get_schema()).validate(json)
    # validate(instance=json, schema=MLHub_schema)


def parse_json(json_path):
    parsed_json = None
    with open(json_path, 'r') as fp:
        parsed_json = json.load(fp)

    validate_json(parsed_json)
    train = parsed_json["train"]
    models = parsed_json["models"]

    features_path, labels_path, category_threshold = parse_train(train)
    parse_models(models, train["split"], features_path,
                 labels_path, category_threshold)


def parse_train(train):
    train_data = train["data"]
    features_path = train_data["features"]
    labels_path = train_data["labels"]
    category_threshold = train_data["category_threshold"]

    verify_exists(features_path)
    verify_exists(labels_path)

    return features_path, labels_path, category_threshold


def parse_model(model, train_split, features_path, labels_path, category_threshold, out_path):
    model_type = model["type"]
    if model_type == 'cnn':
        print("oi")
    else:
        return code_generator.generate_code(
            model, train_split, features_path, labels_path, category_threshold, out_path)
    # else:
    #    raise Exception(f"Model type {model_type} does not exist")


def parse_models(models, train_split, features_path, labels_path, category_threshold):
    out_path = os.path.join("out", str(datetime.time(datetime.now())))
    os.mkdir(out_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(
            parse_model, model, train_split, features_path, labels_path, category_threshold, out_path) for model in models]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    problem_type = models[0]["type"].split('_')[-1]
    code_generator.generate_statistical_comparison(problem_type, out_path)
