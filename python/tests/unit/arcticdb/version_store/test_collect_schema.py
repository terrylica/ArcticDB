"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""

import numpy as np
import pandas as pd
import polars as pl

from arcticdb.options import ArrowOutputStringFormat, OutputFormat
from arcticdb.util.test import assert_frame_equal_with_arrow


def test_collect_schema_basic(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_basic"
    df = pd.DataFrame(
        {"col1": np.arange(1, dtype=np.int64), "col2": [True], "col3": [pd.Timestamp("2025-01-01")]},
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.Int64), ("col2", pl.Boolean), ("col3", pl.Datetime("ns"))])


def test_collect_schema_string_types(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_string_types"
    df = pd.DataFrame(
        {"col1": ["a"], "col2": ["b"]},
    )
    lib.write(sym, df)

    # No overrides
    lazy_df = lib.read(sym, lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.String), ("col2", pl.String)])
    # Default override
    lazy_df = lib.read(sym, arrow_string_format_default=ArrowOutputStringFormat.CATEGORICAL, lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.Categorical(["b"])), ("col2", pl.Categorical(["b"]))])
    # Specific override
    lazy_df = lib.read(sym, arrow_string_format_per_column={"col1": ArrowOutputStringFormat.CATEGORICAL}, lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.Categorical(["b"])), ("col2", pl.String)])
    # Default and specific override
    lazy_df = lib.read(
        sym,
        arrow_string_format_default=ArrowOutputStringFormat.CATEGORICAL,
        arrow_string_format_per_column={"col1": ArrowOutputStringFormat.LARGE_STRING},
        lazy=True,
    )
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.String), ("col2", pl.Categorical(["b"]))])


def test_collect_schema_column_filtering(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_basic"
    df = pd.DataFrame(
        {"col1": np.arange(10, dtype=np.int64), "col2": np.arange(100, 110, dtype=np.float32)},
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, columns=["col2"], lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col2", pl.Float32)])


def test_collect_schema_timeseries(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_timeseries"
    df = pd.DataFrame(
        {"col1": np.arange(10, dtype=np.int64), "col2": np.arange(100, 110, dtype=np.float32)},
        index=pd.date_range("2025-01-01", periods=10),
    )
    # Unnamed, no column selection
    lib.write(sym, df)
    schema = lib.read(sym, lazy=True).collect_schema()
    assert schema == pl.Schema([("index", pl.Datetime("ns")), ("col1", pl.Int64), ("col2", pl.Float32)])
    # Named, no column selection
    df.index.name = "ts"
    lib.write(sym, df)
    schema = lib.read(sym, lazy=True).collect_schema()
    assert schema == pl.Schema([("ts", pl.Datetime("ns")), ("col1", pl.Int64), ("col2", pl.Float32)])
    # With column selection. Note the index column is dropped, as this will be the behaviour Polars expects
    schema = lib.read(sym, columns=["col2"], lazy=True).collect_schema()
    assert schema == pl.Schema([("col2", pl.Float32)])


def test_collect_schema_with_query(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_basic"
    df = pd.DataFrame(
        {"col1": np.arange(10, dtype=np.int64), "col2": np.arange(100, 110, dtype=np.float32)},
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df["new_col"] = 2 * lazy_df["col1"]
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.Int64), ("col2", pl.Float32), ("new_col", pl.Int64)])


def test_collect_schema_and_data(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_and_data"
    df = pd.DataFrame(
        {"col1": np.arange(10, dtype=np.int64), "col2": np.arange(100, 110, dtype=np.float32)},
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    lazy_df.collect_schema()
    received_df = lazy_df.collect().data
    assert_frame_equal_with_arrow(df, received_df)
