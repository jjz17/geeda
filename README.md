# geeda
Pronounced ghee-dah or jee-dee-aye \
Gee.D.A, GEneral EDA: General EDA Framework for Pandas DataFrames

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/jjz17/geeda

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/geeda/).

```bash
# PyPI
pip install pandas
```

# Ideas
* Establish standard output format of functions: e.g. Dict[column, data]
* Add settings to Geeda object for max_categorical_threshold value, etc.

# Design Documentation

## Workflow using Geeda
1. Create the `Geeda()` object by passing in the dataframe for analysis
2. Call the `apply()` function with the desired arguments
3. All validation/cleaning of inputs will be done through `apply()` (with a helper function). This means that all eda functions do not need to convert str to list, make sure columns are in the dataframe, etc


# Poetry Workflow

## Add new dependencies
1. `poetry add <package>`
2. `poetry update`
3. `poetry lock` (Not necessary)

# Technologies Used
* Python3
* Poetry (dependency management, publishing library)
* Docker (containerization of services and tests)
* Git and GitHub (version control)
* Pre-commit (hooks for improved code quality)
* Black (code formatting)
