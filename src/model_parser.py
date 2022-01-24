import json
import os
from datetime import datetime
import concurrent.futures
from jsonschema import Draft7Validator, validators
import code_generator


def run_in_docker(specs_path, features_path, labels_path):

    import docker_layer
    print('Trying to get docker client...', end='')
    client = docker_layer.get_docker_client()

    if client is None:
        print('Failed')
        return
    print('Done')

    container = docker_layer.create_container(client)
    print(f'Created container {container.id}')
    container.start()

    tar_stream = docker_layer.create_tar_stream([
        ('specs.json', open(specs_path, 'rb').read()),
        ('features.csv', open(features_path, 'rb').read()),
        ('labels.csv', open(labels_path, 'rb').read()),
    ])

    print('Moving files to container...', end='')
    container.put_archive('/home/mluser/', tar_stream)
    print('Done')

    print('Will now execute the model(s)')
    res = container.exec_run(
        'python3 src/main.py --specs /home/mluser/specs.json --features /home/mluser/features.csv --labels /home/mluser/labels.csv', user='mluser', workdir='/home/mluser')

    print(res.output.decode())
    print(f'Will now kill container {container.id}...', end='')
    container.kill()
    print('Done')
    print(f'Will now stop container {container.id}...', end='')
    container.stop()
    print('Done')


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if not isinstance(subschema, bool) and "default" in subschema:
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
    validator = DefaultValidatingDraft7Validator(get_schema())
    if not validator.is_valid(json):
        errors = sorted(validator.iter_errors(json), key=lambda e: e.path)
        for error in errors:
            print(error.message)

        exit(1)


def parse_json(json_path, dockerize=False, features=None, labels=None):
    parsed_json = None
    with open(json_path, 'r') as fp:
        parsed_json = json.load(fp)

    validate_json(parsed_json)
    train = parsed_json["train"]
    models = parsed_json["models"]

    features_path, labels_path = parse_train(train, features_path_exists=(
        features is not None), labels_path_exists=(labels is not None))

    if features is not None:
        features_path = features

    if labels is not None:
        labels_path = labels

    if dockerize:
        run_in_docker(json_path, features_path, labels_path)
    else:
        parse_models(models, train, features_path, labels_path)


def parse_train(train, features_path_exists=False, labels_path_exists=False):
    train_data = train["data"]
    features_path = train_data["features"]
    labels_path = train_data["labels"]

    if not features_path_exists:
        verify_exists(features_path)
    if not labels_path_exists:
        verify_exists(labels_path)

    return features_path, labels_path


def parse_model(model, train, features_path, labels_path, out_path):
    model_type = model["type"]
    if model_type == 'cnn':
        return code_generator.generate_cnn(model, train, features_path, labels_path, out_path)
    else:
        return code_generator.generate_code(
            model, train, features_path, labels_path, out_path)


def parse_models(models, train, features_path, labels_path):
    out_path = os.path.join("out", str(datetime.time(datetime.now())))
    os.makedirs(out_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(
            parse_model, model, train, features_path, labels_path, out_path) for model in models]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    if len(models) > 1:
        problem_type = models[0]["type"].split('_')[-1]
        code_generator.generate_statistical_comparison(problem_type, out_path)
