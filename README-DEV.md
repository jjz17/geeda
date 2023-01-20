# Design Documentation

## Workflow using Geeda
1. Create the `Geeda()` object by passing in the dataframe for analysis
2. Call the `apply()` function with the desired arguments
3. All validation/cleaning of inputs will be done through `apply()` (with a helper function). This means that all eda functions do not need to convert str to list, make sure columns are in the dataframe, etc

# Poetry Workflow

## Add new dependencies
1. `poetry add <package>`
2. `poetry update`
3. `poetry lock --no-update` (Not necessary)

# Ideas
* Establish standard output format of functions: e.g. Dict[column, data]
* Add settings to Geeda object for max_categorical_threshold value, etc.
* Add CI/CD with Github or Jenkins
