# Schema

Let's take a look at our schema so you can better understand the different sections required in the input file and how you can customize them

## Train

The first section required by our tool is the *train* section. In this section a set of information regarding the dataset and the configuration of the training should be specified.

**Usage**:

```
"train": {
    "split": {
        "test_size": TEST_SIZE,
        "train_size": TRAIN_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": SHUFFLE,
        "stratify": STRATIFY
    },
    "data": {
        "features": FEATURES,
        "category_threshold": CATEGORY_TRESHOLD,
        "labels": LABELS,
        "num_classes": NUM_CLASSES
    },
    "train_settings": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT
    }
}
```

- ``TEST_SIZE``: a number between 0 and 1 that represents the proportion of the dataset to include in the test split. 
    - **Default**: ``0.2``
- ``TRAIN_SIZE``: a number between 0 and 1 that represents the proportion of the dataset to include in the train split. 
    - **Default**: ``null``
- ``RANDOM_STATE``: an integer to control the shuffling applied to the data before applying the split. 
    - **Default**: ``null``
- ``SHUFFLE``: a boolean to specify whether or not to shuffle the data before splitting. 
    - **Default**: ```true```
- ``STRATIFY``: a string to specify whether the data is split in a stratified fashion.
    - **Possible values**: `` "features", "labels"``
    - **Default**: ``null``
- ``FEATURES``: a string to specify the path to the file containing the dataset features. 
    - **Required**
- ``CATEGORY_TRESHOLD``: an integer that represents the treshold that will be used to create or not a category. 
    - **Default**: ``10``
- ``LABELS``: a string to specify the path to the file containing the dataset labels.
    - **Required**
- ``NUM_CLASSES``: an integer to specify the number of different classes in the label's file (use for image classification). 
    - **Default**: ``null``
- ``EPOCHS``: an integer that represents the number of epochs to train the model (use for image classification). 
    - **Default**: ``10``
- ``BATCH_SIZE``: an integer representing the number of samples per gradient update (use for image classification). 
    - **Default**: ``null``
- ``VALIDATION_SPLIT``: a number between 0 and 1 to specify the fraction of the training data to be used as validation data (use for image classification). 
    - **Default**: ```0```


Here is a small example of the train section used for comparing two models in image classification:
```json=
"train": {
    "split": {
      "test_size": 0.1,
      "random_state": 0
    },
    "data": {
      "features": "/home/vitor/Documents/TACS/MLhub/data/cnn/features.csv",
      "labels": "/home/vitor/Documents/TACS/MLhub/data/cnn/labels.csv"
    },
    "train_settings": {
      "epochs": 2,
      "batch_size": 128 ,
      "validation_split": 0.1
    }
}
```

## Models

In the *models* section you must specify which models you want to compare. This way you must specify the different models you want to use in an array. Different models can have different properties.

**Usage**:

```
"models": [
    MODEL1,
    MODEL2
]
```

Here is a small example of the model section used for comparing classification models:

```json=
"models": [
    {
        "type": "decision_tree_classification",
        "name": "tree classifier"
    },
    {
        "type": "decision_tree_classification",
        "name": "tree2 classifier",
        "criterion": "entropy"
    },
    {
        "type": "support_vector_machine_classification",
        "name": "svm classifier"
    },
    {
        "type": "k_nearest_neighbours_classification",
        "name": "knn classifier"
    },
    {
        "type": "random_forest_classification",
        "name": "forest classifier"
    },
    {
        "type": "naive_bayes_classification",
        "name": "bayes classifier"
    }
]
```

### Regression

Each model has a different set of properties that you can use. For regression models you can find their different properties in the following sections

#### Decision Tree
The properties of this model were created taking into account the specification of the [SKLearn Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "decision_tree_regression"
        "name": NAME,
        "criterion": CRITERION,
        "splitter": SPLITTER,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_weight_fraction_leaf": MIN_WEIGHT_FRACTION_LEAF,
        "max_features": MAX_FEATURE,
        "random_state": RANDOM_STATE,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE,
        "ccp_alpha": CPP_ALPHA
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``

#### Linear
The properties of this model were created taking into account the specification of the [SKLearn Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "linear_regression"
        "name": NAME,
        "fit_intercept": FIT_INTERCEPT,
        "copy_X": COPY_X,
        "n_jobs": N_JOBS,
        "positive": POSITIVE
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``

#### Polynomial
The properties of this model were created taking into account the specification of the [SKLearn Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) class and the [SKLearn Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html). 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "polynomial_regression"
        "name": NAME,
        "fit_intercept": FIT_INTERCEPT,
        "copy_X": COPY_X,
        "n_jobs": N_JOBS,
        "positive": POSITIVE,
        "degree": DEGREE,
        "interaction_only": INTERACTION_ONLY,
        "include_bias": INCLUDE_BIAS,
        "order": ORDER
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``


#### Random Forest
The properties of this model were created taking into account the specification of the [SKLearn Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "random_forest_regression"
        "name": NAME,
        "n_estimators": N_ESTIMATORS,
        "criterion": CRITERION,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_weight_fraction_leaf": MIN_WEIGHT_FRACTION_LEAF,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE,
        "ccp_alpha": CPP_ALPHA,
        "bootstrap": BOOTSTRAP,
        "oob_score": OOB_SCORE,
        "n_jobs": N_JOBS,
        "verbose": VERBOSE,
        "warm_start": WARM_START,
        "max_samples": MAX_SAMPLES
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``

#### Support Vector
The properties of this model were created taking into account the specification of the [SKLearn Support Vector Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "support_vector_regression"
        "name": NAME,
        "kernel": KERNEL,
        "degree": DEGREE,
        "gamma": GAMMA,
        "coef0": COEF0,
        "tol": TOL,
        "C": C,
        "epsilon": EPSILON,
        "shrinking": SHRINKING,
        "cache_size": CACHE_SIZE,
        "verbose": VERBOSE,
        "max_iter": MAX_ITER
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``



### Classification

Each model has a different set of properties that you can use. For classification models you can find their different properties in the following sections

#### Decision Tree
The properties of this model were created taking into account the specification of the [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) class. 
For more information regarding these properties visit the class page.


```
"models": [
    {
        "type": "decision_tree_classification"
        "name": NAME,
        "criterion": CRITERION,
        "splitter": SPLITTER,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_weight_fraction_leaf": MIN_WEIGHT_FRACTION,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE,
        "ccp_alpha": CCP_LAPHA
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``


#### K-Nearest Neighbors
The properties of this model were created taking into account the specification of the [K-nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "k_nearest_neighbor_classification"
        "name": NAME,
        "n_neighbors": N_NEIGHBORS,
        "weights": WEIGHTS,
        "algorithm": ALGORITHM,
        "leaf_size": LEAF_SIZE,
        "p": P,
        "metric": METRIC,
        "coef0": COEF0,
        "n_jobs": N_JOBS

    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``
    
#### Gaussian Naive Bayes
The properties of this model were created taking into account the specification of the [Gaussian Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "gaussian_naive_bayes_classification"
        "name": NAME,
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``

#### Random Forest
The properties of this model were created taking into account the specification of the [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "random_forest_classification"
        "name": NAME,
        "n_estimators": N_ESTIMATORS,
        "criterion": CRITERION,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_weight_fraction_leaf": MIN_WEIGHT_FRACTION_LEAF,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE,
        "ccp_alpha": CPP_ALPHA,

    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``

#### Support Vector Machines
The properties of this model were created taking into account the specification of the [Support Vector](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class. 
For more information regarding these properties visit the class page.

```
"models": [
    {
        "type": "support_vector_machine_classification"
        "name": NAME,
        "C": C,
        "kernel": KERNEL,
        "degree": DEGREE,
        "gamma": GAMMA,
        "coef0": COEF0,
        "tol": TOL,
        "shrinking": SHRINKING,
        "probability": PROBABILITY,
        "max_iter": MAX_ITER,
        "decision_function_shape": DECISION_FUNCTION_SHAPE,
        "break_ties": BREAK_TIES
    }
]
```

- ``NAME``: a string that will be used to identify the model. 
    - **Default**: ``null``


### Image Classification

For image classification it is necessary to have some specific properties in the model in order to specify the different layers, the loss function that should be used, etc.

**Usage**:

```
"models": [
    {
        "type": "cnn",
        "loss": LOSS,
        "optimizer": {
            "name": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
        "input": {
            "type": "images", 
            "shape": SHAPE
        },
        "layers": [
            LAYER1,
            LAYER2
        ]
    }
]
```

- ``LOSS``: a string that represents the loss function to use during training. 
    - **Required**
    - **Possible values**: ``"categorical_crossentropy", "mean_squared_error"``
- ``OPTIMIZER_NAME``: a string that represents the optimizer to use during training. 
    - **Possible values**: ``"adam", "adadelta", "sgd", "rmsprop"``
    - **Default**: ``adam``
- ``LEARNING_RATE``: a number to control the learning rate used by the optimizer. 
    - **Default**: ``null``
- ``SHAPE``: a boolean to specify whether or not to shuffle the data before splitting. 
    - **Required**


#### Layers
For the different layers you need to specify a different set of properties. In the next sections you can find the properties needed for each layer

##### Flatten
The flatten layer does not need any properties this way you just need to enter the following to add it.

**Usage**:

```
"layers": [
    ...
    "flatten",
    ...
]
```

##### Dense

**Usage**:

```
"layers": [
    {
        "dense": {
            "units": UNITS,
            "activation": ACTIVATION
        }
    }
]
```

- ``UNITS``: an integer that indicates the number of nodes in this dense layer. 
    - **Required**
- ``ACTIVATION``: a string that represents the activation function to use in this layer. 
    - **Possible values**: ``"softmax", "relu", "tanh", "sigmoid"``
    - **Default**: ``null``

##### Dropout

**Usage**:

```
"layers": [
    {
        "dropout": DROPOUT
    }
]
```

- ``DROPOUT``: a number between 0 and 1 that indicates the fraction of activations to drop at each pass. 

##### Convolution

**Usage**:

```
"layers": [
    {
        "convolution": {
            "size": SIZE,
            "filters": FILTERS,
            "activation": ACTIVATION
        }
    }
]
```

- ``SIZE``: an integer (for 1-D convolutions) or a list of integers which specify the size of the receptive field of the convolution. 
    - **Required**
- ``FILTERS``: an integer that represents the number of convolutional filters to apply. 
    - **Required**
- ``ACTIVATION``: a string that represents the activation function to use in this layer. 
    - **Possible values**: ``"softmax", "relu", "tanh", "sigmoid"``
    - **Default**: ``null``

##### Activation

**Usage**:

```
"layers": [
    {
        "activation": ACTIVATION
    }
]
```

- ``ACTIVATION``: a string that represents the activation function to use in this layer. 
    - **Possible values**: ``"softmax", "relu", "tanh", "sigmoid"``

##### Pooling

**Usage**:

```
"layers": [
    {
        "pooling": {
            "size": SIZE,
            "type": TYPE,
        }
    }
]
```

- ``SIZE``: an integer (for 1-D pools) or a list of integers which specify the size of the pooling layer's receptive field. 
    - **Required**
- ``TYPE``: a string that represents the type of pooling to apply. 
    - **Required**
    - **Possible values**: ``"max", "average"``

##### Batch Normalization

**Usage**:

```
"layers": [
    {
        "batch_normalization": {
            "axis": AXIS
        }
    }
]
```

- ``AXIS``: an integer indicating which axis to apply the normalization to. 
    - **DEFAULT**: ``-1``



Here is a small example of the models section used for comparing two models in image classification:
```json=
"models": [
    {
        "type": "cnn",
        "loss": "categorical_crossentropy",
        "input": {"type": "images", "shape": [28,28]},
        "layers": [
            {
              "convolution": {
                "filters": 32,
                "size": [3, 3],
                "activation":"relu"
              }
            },
            {
              "pooling": {
                "size": [2, 2],
                "type": "max"
              }
            },
            "flatten",
            {
              "dense": {
                "units": 128,
                "activation":"relu"
              }
            },
            {
              "dense": {
                "units": 10,
                "activation":"softmax"
              }
            }
        ]
    }
  ]
```
