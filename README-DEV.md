# Design Documentation
1. All validation/cleaning of inputs will be done through `apply()` (with a helper function). This means that EDA functions do not need to convert str to list, make sure columns are in the dataframe, etc.

# Poetry Workflow

## Add new dependencies
1. `poetry add <package>`
2. `poetry update`
3. `poetry lock --no-update` (Not necessary)

# Semantic Versioning

Version number is maintained via the version in `pyproject.toml`. \
Version format: MAJOR.MINOR.PATCH

1. PATCH increments for each bug fix merged into `main`
2. MINOR increments with each significant feature addition
3. MAJOR increments with the introduction of a  breaking change to the API

# Ideas
* Establish standard output format of functions: e.g. Dict[column, data]
* Add settings to Geeda object for max_categorical_threshold value, etc.
* Add CI/CD with Github or Jenkins
