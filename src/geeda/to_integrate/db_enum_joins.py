# COPYRIGHTS AND PERMISSIONS:
# Copyright 2022 MORSECORP, Inc. All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Dict, List, Tuple, Any
import pandas as pd


def get_enum_to_value_mappings(
    main_df: pd.DataFrame, **dataframes: pd.DataFrame
) -> Dict[Any, Tuple[str, Any]]:
    """
    Get mapping of enum columns in the main dataframe to value columns from the other dataframes.

    :param main_df:
        main dataframe to replace enum values in
    :param dataframes:
        enum values dataframes to extract replacement data from
    :return:
        A dict with keys being the enum column names and values being tuples with the first element being the name
        of the dataframe containing the values of the corresponding enum and the second element being the column name
        that contains the value data.
    """
    # Classify main_df's columns as enums or values
    value_columns = list(main_df.columns)
    enum_to_value = dict()
    for name, df in dataframes.items():
        for column in df.columns:
            # If column is in main and another df, it is an enum
            if column in main_df.columns:
                if column in value_columns:
                    value_columns.remove(column)
                # Map the df's other column as the value pair to the enum, raise error if more than 1 other column
                if len(df.columns) == 2:
                    enum_to_value[column] = (
                        name,
                        list(filter(lambda x: x != column, df.columns))[0],
                    )
                else:
                    raise ValueError(
                        f"Table: {name} contains the enum `{column}` but should only have 2 columns,"
                        f" an enum and a value column. It currently has {len(df.columns)} columns."
                    )
    return enum_to_value


def get_enum_joins_query(
    main_table_name: str,
    main_table_columns: List[str],
    enum_to_value_mapping: Dict[str, Tuple[str, str]],
) -> str:
    """
    Get a SQL query that replaces the enums in the main table with values provided by the given enum tables.

    :param main_table_name:
        Name of the main table
    :param main_table_columns:
        List of the main table's column names
    :param enum_to_value_mapping:
        A dict with keys being the enum column names and values being tuples with the first element being the name
        of the table containing the values of the corresponding enum and the second element being the column name
        that contains the value data.
    :return:
        The SQL query that replaces enums with values in the main table
    """
    enum_columns = set(enum_to_value_mapping.keys())
    main_table_enum_columns = set(main_table_columns) & enum_columns
    main_table_value_columns = set(main_table_columns) - main_table_enum_columns

    # Sort column lists
    sorted_main_table_enum_columns = sorted(main_table_enum_columns)
    sorted_main_table_value_columns = sorted(main_table_value_columns)

    selects = []
    joins = []

    for enum in sorted_main_table_enum_columns:
        table_name, value = enum_to_value_mapping[enum]
        selects.append(f"{table_name}.{value}")
        joins.append(
            f"JOIN {table_name} ON {table_name}.{enum} = {main_table_name}.{enum}"
        )

    for value_column in sorted_main_table_value_columns:
        selects.append(f"{main_table_name}.{value_column}")

    select_statement = ", ".join(selects)
    join_statement = " ".join(joins)
    sql_query = f"SELECT {select_statement} FROM {main_table_name} {join_statement}"

    return sql_query


def db_enum_joins(
    main_df: pd.DataFrame,
    main_df_name: str,
    printout: bool = True,
    **dataframes: pd.DataFrame,
) -> str:
    """
    Replace enum values in the main dataframe with values from the other dataframes.

    :param main_df:
        main dataframe to replace enum values in
    :param main_df_name:
        table name of the main dataframe
    :param printout:
        True if a printout of the SQL query is desired
    :param dataframes:
        enum values dataframes to extract replacement data from
    :return:
        The MySQL query string to replace enum values in the main dataframe
    """
    # Classify main_df's columns as enums or values
    value_columns = list(main_df.columns)
    enum_to_value = dict()
    for name, df in dataframes.items():
        for column in df.columns:
            # If column is in main and another df, it is an enum
            if column in main_df.columns:
                value_columns.remove(column)
                # Map the df's other column as the value pair to the enum, raise error if more than 1 other column
                if len(df.columns) == 2:
                    enum_to_value[column] = [
                        name,
                        list(filter(lambda x: x != column, df.columns))[0],
                    ]
                else:
                    raise ValueError(
                        f"Table: {name} should only have 2 columns, an enum and a value column"
                    )
    selects = []
    joins = []
    for enum, (name, value) in enum_to_value.items():
        selects.append(f"{name}.{value}")
        joins.append(f"JOIN {name} ON {name}.{enum} = {main_df_name}.{enum}")
    for value_column in value_columns:
        selects.append(f"{main_df_name}.{value_column}")

    select_statements = ", ".join(selects)
    join_statements = " ".join(joins)
    sql_query = f"SELECT {select_statements} FROM {main_df_name} {join_statements}"

    if printout:
        # Print query in readable format
        print("SELECT")
        for i, select in enumerate(selects):
            if i != len(selects) - 1:
                print(f"\t{select},")
            else:
                print(f"\t{select}")
        print(f"FROM\n\t{main_df_name}")
        for join in joins:
            print(join)

    return sql_query
