import json
import os
from jsonschema import validate
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


DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)

MLHub_schema = {
    "required": ["train"],
    "type": "object",
    "properties": {
        "train": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "string"
                        },
                        "labels": {
                            "type": "string"
                        }
                    }
                }
            }
        },
        "model": {
            "type": "object",
            "properties": {
                "type": {
                    "default": "decision_tree_regression",
                    "enum": ["decision_tree_regression", "linear_regression", "polynomial_regression", "random_forest_regression", "support_vector_regression"]
                }
            },
            "allOf": [
                {
                    "if": {
                        "properties": {"type": {"const": "decision_tree_regression"}}
                    },
                    "then": {
                        "properties": {
                            "criterion": {
                                "default": "squared_error",
                                "enum": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                            },
                            "splitter": {
                                "default": "best",
                                "enum": ["best", "random"]
                            },
                            "max_depth": {"type": ["integer", "null"], "default": None},
                            "min_samples_split": {"type": "number", "default": 2},
                            "min_samples_leaf": {"type": "number", "default": 1},
                            "min_weight_fraction_leaf": {"type": "number", "default": 0.0},
                            "max_features": {
                                "type": ["integer", "string", "null"],
                                "default": None
                            },
                            "random_state": {"type": ["integer", "null"], "default": None},
                            "max_leaf_nodes": {"type": ["integer", "null"], "default": None},
                            "min_impurity_decrease": {"type": "number", "default": 0.0},
                            "ccp_alpha": {"type": "number", "default": 0.0, "minimum": 0},
                        }
                    }
                },
            ]
        },
    }
}


def verify_exists(path):
    if not os.path.exists(path):
        raise Exception(f"File {path} was not found")


def validate_json(json):
    DefaultValidatingDraft7Validator(MLHub_schema).validate(json)
    print(json)
    #validate(instance=json, schema=MLHub_schema)


def parse_json(json_path):
    parsed_json = None
    with open(json_path, 'r') as fp:
        parsed_json = json.load(fp)

    validate_json(parsed_json)
    train = parsed_json["train"]
    model = parsed_json["model"]

    features_path, labels_path = parse_train(train)
    parse_model(model, features_path, labels_path)


def parse_train(train):
    train_data = train["data"]
    features_path = train_data["features"]
    labels_path = train_data["labels"]

    verify_exists(features_path)
    verify_exists(labels_path)

    return features_path, labels_path


def parse_model(model, features_path, labels_path):
    model_type = model["type"]
    if model_type == 'linear_regression':
        code_generator.linear_regression_generator(features_path, labels_path)
    elif model_type == 'polynomial_regression':
        code_generator.polynomial_regression_generator(
            features_path, labels_path)
    elif model_type == 'decision_tree_regression':
        # print(model["criterion"])
        code_generator.decision_tree_generator(features_path, labels_path)
    elif model_type == 'support_vector_regression':
        code_generator.support_vector_generator(features_path, labels_path)
    elif model_type == 'random_forest_regression':
        code_generator.random_forest_generator(features_path, labels_path)
    elif model_type == 'cnn':
        print("oi")
    else:
        raise Exception(f"Model type {model_type} does not exist")
