"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""

import numpy as np
import pandas as pd
import polars as pl

from arcticdb.options import OutputFormat


def test_collect_schema_basic(lmdb_library):
    lib = lmdb_library
    lib._nvs.set_output_format(OutputFormat.POLARS)
    sym = "test_collect_schema_basic"
    df = pd.DataFrame(
        {"col1": np.arange(10, dtype=np.int64), "col2": np.arange(100, 110, dtype=np.float32)},
    )
    lib.write(sym, df)

    lazy_df = lib.read(sym, lazy=True)
    schema = lazy_df.collect_schema()
    assert schema == pl.Schema([("col1", pl.Int64), ("col2", pl.Float32)])
