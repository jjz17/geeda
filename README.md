# geeda
Pronounced ghee-dah or gee-dee-aye \
Gee.D.A, GEneral EDA: General EDA Framework for Pandas DataFrames


# Design Documentation

## Workflow using Geeda
1. Create the `Geeda()` object by passing in the dataframe for analysis
2. Call the `apply()` function with the desired arguments
3. All validation/cleaning of inputs will be done through `apply()` (with a helper function). This means that all eda functions do not need to convert str to list, make sure columns are in the dataframe, etc

