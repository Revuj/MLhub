# Examples: In Depth

Let's look at some examples of how fun and easy MLhub makes state-of-the-art machine learning



## Profit Prediction (Regression)

Let's jump right in and see how awesome MLhub is! The first example we'll look at is a [dataset](https://github.com/Revuj/MLhub/tree/master/data/regression) with KPIs about companies. 
The goal here is to predict the anual profit of each company.


First, you need [install MLHub](https://github.com/Revuj/MLhub#installation) like so:

> Clone the repository

    $ git clone git@github.com:Revuj/MLhub.git
    

> Setup a virtual environment

    $ python3 -m venv env

> Install the requirements


    $ pip install -r requirements.txt



Now we have to define our own regression model in a JSON file. Let's use [this example](https://github.com/Revuj/MLhub/blob/master/examples/regression/decision_tree_regression.json) with a decision tree model:

```json
{
  "train": {
    "split": {
      "test_size": 0.1,
      "random_state": 0
    },
    "data": {
      "features": "/Users/rafavarela/Projects/MLhub/data/regression/features.csv",
      "labels": "/Users/rafavarela/Projects/MLhub/data/regression/labels.csv"
    }
  },
  "models": [
    {
      "type": "decision_tree_regression",
      "criterion": "absolute_error",
      "splitter": "best",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": null,
      "random_state": null,
      "max_leaf_nodes": 3,
      "min_impurity_decrease": 15.1,
      "ccp_alpha": 0.0
    }
  ]
}
```

As you can see, in this case we are using 10% of the dataset for test purposes and we are also overwritting some of the decision tree models parameters.

After this, we are ready to run MLhub and train our model!

    $ python3 src/main.py --specs examples/regression/decision_tree_regression.json --dockerize
    
You can see that we have set the *dockerize flag* to true. This means that our program will be run inside a Docker container.
Here's the shell output:

```
Executing python notebook...
Executed decision_tree_regression_ab72b314-7d27-11ec-> b009-b62bca79ceda model
```


What just happened? MLhub parsed our model, and generated and executed a report than trains the model and evaluates its performance.

After this is over, you can now check the [*out* folder](https://github.com/Revuj/MLhub/tree/master/examples_out/regression). There you can see a csv file with statistics about the model, the resulting parsed notebook and also the executed notebook where you can see information about the training procewss and testing results.  

![](https://i.imgur.com/vXHw5lM.png)

![](https://i.imgur.com/zkRQPkG.png)

Hope you learned and had fun!

## Product Purchase Prediction (Classifcation)

This time we have a classification [problem](https://github.com/Revuj/MLhub/tree/master/data/classification) about people. 
The goal here is to predict if a person is or isn't buying a specific product.

Let's also try to do a comparison between different classification models. [Here's how it's done](https://github.com/Revuj/MLhub/blob/master/examples/classification/multiple_models_classification.json):

```json
{
    "train": {
      "split": {
        "test_size": 0.1,
        "random_state": 0
      },
      "data": {
        "features": "/home/krystal/MLhub/data/classification/features.csv",
        "labels": "/home/krystal/MLhub/data/classification/labels.csv"
      }
    },
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
  }
  
   
```

We are defining 6 different classification models, so the results will be a bit different this time. We can run the program like so:

    $ python3 src/main.py --specs examples/classification/multiple_models_classification.json 

Here's the output:

```
Executing python notebook...
Executing python notebook...
Executing python notebook...
Executing python notebook...
Executing python notebook...
Executing python notebook...
Executed tree classifier model
Executed knn classifier model
Executed svm classifier model
Executed tree2 classifier model
Executed bayes classifier model
Executed forest classifier model
Executing statistical comparison python notebook...
Executed statistical comparison
```

You can see that this time a statistical comparison report was generated in the [*out* folder](https://github.com/Revuj/MLhub/tree/master/examples_out/classification). But first, let's look at a report for one of the models:

![](https://i.imgur.com/fRUfy5q.png)


Since we have more than one model, you can also take a look at the statistical comparison report, which gives you an overview about all model's metrics:

![](https://i.imgur.com/gSxCopf.png)

![](https://i.imgur.com/QX3455H.png)

## Fashion MNIST (Image Classifcation)

This time we have an image classification [problem](https://github.com/Revuj/MLhub/tree/master/data/cnn) about clothes. 
The goal here is to predict the type of clothing is represented in an image.

Let's also try to do a comparison between different CNN models. One without using Dropout layers and other using them [Here's how it's done](https://github.com/Revuj/MLhub/blob/master/examples/cnn.json):


```json
{
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
  },
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
        {
          "convolution": {
            "filters": 64,
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
        {
          "convolution": {
            "filters": 128,
            "size": [3, 3],
            "activation":"relu"
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
    },
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
        {"dropout": 0.25},
        {
          "convolution": {
            "filters": 64,
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
        {"dropout": 0.25},
        {
          "convolution": {
            "filters": 128,
            "size": [3, 3],
            "activation":"relu"
          }
        },
        {"dropout": 0.4},
        "flatten",
        {
          "dense": {
            "units": 128,
            "activation":"relu"
          }
        },
        {"dropout": 0.3},
        {
          "dense": {
            "units": 10,
            "activation":"softmax"
          }
        }
      ]
    }
  ]
}
```

We are defining 2 different CNN models, so the results will be a comparison between the two models. We can run the program like so:

    $ python3 src/main.py --specs examples/cnn.json  

Here's the output:

```
Executing python notebook...
Executing python notebook...
[...]
Executed cnn_711033ec-7d48-11ec-b087-d7c57270a0f7 model
Executed cnn_71103630-7d48-11ec-b088-d7c57270a0f7 model
Executing statistical comparison python notebook...
Executed statistical comparison
```

You can see that the statistical comparison report was generated in the [*out* folder](https://github.com/Revuj/MLhub/tree/master/examples_out/cnn). But first, let's look at a report for one of the models:

![](https://i.imgur.com/4jDOOGp.png)


![](https://i.imgur.com/qLhqbj3.png)


Since we have more than one model, you can also take a look at the statistical comparison report, which gives you an overview about all model's metrics. Here we can see that using dropout layers increased the performance even though he only trained for 2 epochs.

![](https://i.imgur.com/eRtkDM1.png)

![](https://i.imgur.com/f3JJ7H4.png)

