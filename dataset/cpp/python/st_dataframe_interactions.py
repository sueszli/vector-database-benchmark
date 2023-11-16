# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import pandas as pd

import streamlit as st

np.random.seed(0)
random.seed(0)

random_df = pd.DataFrame(
    np.random.randn(5, 5),
    columns=["Column A", "Column B", "Column C", "Column D", "Column E"],
)

st.dataframe(random_df)

st.data_editor(random_df, num_rows="dynamic")
