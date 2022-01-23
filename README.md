# MLhub

![](https://i.imgur.com/7rJCCew.png)

> Your one-stop hub for all Machine Learning related affairs 😍

*MLhub* is a low code machine learning tool that allows you to automatically train and benchmark machine learning modes without typing a single line of code.

## Index

- [Installation](#installation)
- [Usage](#usage)
- [Background](#background)
- [Features](#features)
- [Architecture](#architecture)
- [Future Work](#future-work)
- [Team](#team)
- [License](#license)

## Installation
> Clone the repository

```git clone git@github.com:Revuj/MLhub.git```

> Install the requirements
>
```pip install -r requirements.txt```

## Usage

```python3 src/main.py --specs <models definition path> [--dockerize dockerize flag]```

* **models definition path** - path to the JSON file containing the definition of the models to be executed.
* **dockerize flag** - boolean that enables a dockerized execution for the application (default is *False*)


## Background

Machine Learning is now more present than ever in our society. It completly changed the game for various industry sectors, and if you're not making any use of it, you're probably going to stay behind.

However, machine learning isn't easy to understand and hard to use as writting code for it can be a pain in the ass, but ideally, **everyone should be able to benefit from this type of techonology, even if you don't know how to write a single line of code**!

This is where *MLhub* comes into place.


## Features

### Low code machine learning

*MLhub* allows you to train multiple machine learning models, for regression and classification problems, without writing a single line of code. In fact, the only thing you need is a JSON file where you define the location of your datasets and the models that you want to test. Then you just need to wait for some *MLhub* magic happening behind the scenes ✨.

<img src="https://i.imgur.com/2msYyNe.png" width="400"/>



### Auto Machine learning

*MLhub* gives you the freedom to fully customize your models, in case you know what you're actually doing, or you can also just pick your model and leave the rest to us (like in the example above)! 

If you want to costumize them, you should be able to use any parameter that is used in the [sklearn](https://scikit-learn.org/stable/index.html) models. Here's an example for a [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html):

<img src="https://i.imgur.com/asFbFPz.png" width="400"/>


### Train and Benchmarks models

Our tool allows you to automatically train models and provide you results that allow you to evaluate their performance. For **regression problems** we measure metrics such as *r2 score*, *mean squared error* and *mean absolute error*. For **classification problems** we measure metrics such as *accuracy*, *precision*, *recall* and *f1 score*. In both cases we also measure the *CPU time* so that you also have some information about performance.


### Report Generation

For each model that you have defined on your JSON file, we provide you two *python notebooks* (one that was executed already and one that wasn't), with everything that we ran behind the scenes, so you can understand what happened and modify something if you want. 

<img src="https://i.imgur.com/lgOX0JG.png" width="300"/>

 

We also generate a CSV file and special report that hold a statistical comparison between the different models.

<img src="https://i.imgur.com/4LppTXN.png" width="300"/>

## Architecture
![](https://i.imgur.com/3mUU0Sw.png)

The program is executed through the shell using just one argument which is the path for the JSON file containing the data paths and models' definitions.



### Parser

This module is responsible for parsing and validating the JSON file. This is done using the python [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/) library. After that, it guarantees asynchronous execution by creating a pool of processes (one for each module) and then passing it the necessary information for the *Generator* module.

### Generator

Constructs and executes reports in the form of *csv* and *python notebook* files. Since most notebooks have a significant amount of common code, we utilized the [jinja2](https://pypi.org/project/Jinja2/) template engine and its inheritance capabilities as a way to render the final notebooks efficiently and with style. All of the notebooks are executed using [nbconvert preprocessors](https://nbconvert.readthedocs.io/en/latest/api/preprocessors.html).

Both the *Parser* and *Generator* modules can be executed inside a [Docker container](https://www.docker.com/resources/what-container) so that our application runs smoothly, quickly and securely in any system.

## In the context of TACS (Advanced Software Construction Techniques)

We believe this project fits the TACS course perfectly as it represents a nice example of **Metaprogramming and Model-Driven Engineering** for two different reasons:
* we have made use of a DSL (**Domain Specific Language**) in order to define the different models and its parameters using a JSON file
* our tool performs **source-code generation** to construct the different reports that are responsible for training and evaluating machine learning models. 
 
MLhub can also be categorized as an **Automatic Programming** tool as it is capable of synthesizing machine learning programs with just a few simple inputs that anyone is capable of writting, even those with very limited machine learning knowledge. 

## Future Work

* Support more models (e.g. neural networks).
* Support more types of problems (e.g. image classification).
* Have more Auto ML features (e.g. perform grid search to find the best parameters for each model).
* Build a user interface to provide an alternative for composing the JSON input file.


## Team

* Rafa Varela
* José Silva
* Vítor Barbosa


## License

The project is MIT licensed.
