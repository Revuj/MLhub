import json
import os
from jsonschema import validate
import code_generator

MLHub_schema = {
    "required": ["train"],
    "type": "object",
    "properties": {
        "train" : {
            "type": "object",
            "properties": {
              "data" : {
                "type": "object",
                "properties": {
                  "features" :{
                    "type": "string"
                  },
                  "labels" :{
                    "type": "string"
                  }
                }
              }
            }
        },
        "model" : {
            "type": "object",
            "properties": {
              "type" : {
                "type": "string",
              }
            }
        },
    }
}


def verify_exists(path):
  if not os.path.exists(path):
      raise Exception(f"File {path} was not found")


def validate_json(json):
    validate(instance=json, schema=MLHub_schema)


def parse_json(json_path):
  parsed_json = None
  with open(json_path ,'r') as fp:
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
  type = model["type"]
  if type =='linear_regression':
    code_generator.linear_regression_generator(features_path, labels_path)
  elif type =='decision_tree_regression':
    print("oi")
  elif type =='random_forest_regression':
    print("oi")
  elif type =='support_vector_regression':
    print("oi")
  elif type =='cnn':
    print("oi")
  else:
    raise Exception(f"Model type {type} does not exist")
