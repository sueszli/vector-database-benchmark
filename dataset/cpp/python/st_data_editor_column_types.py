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

import streamlit as st
from tests.streamlit.data_mocks import (
    BASE_TYPES_DF,
    DATETIME_TYPES_DF,
    INTERVAL_TYPES_DF,
    LIST_TYPES_DF,
    NUMBER_TYPES_DF,
    PERIOD_TYPES_DF,
    SPECIAL_TYPES_DF,
    UNSUPPORTED_TYPES_DF,
)

np.random.seed(0)
random.seed(0)

st.set_page_config(layout="wide")

st.subheader("Base types")
st.data_editor(BASE_TYPES_DF, use_container_width=True)

st.subheader("Number types")
st.data_editor(NUMBER_TYPES_DF, use_container_width=True)

st.subheader("Date, time and datetime types")
st.data_editor(DATETIME_TYPES_DF, use_container_width=True)

st.subheader("List types")
st.data_editor(LIST_TYPES_DF, use_container_width=True)

st.subheader("Interval dtypes in pd.DataFrame")
st.data_editor(INTERVAL_TYPES_DF, use_container_width=True)

st.subheader("Special types")
st.data_editor(
    SPECIAL_TYPES_DF,
    use_container_width=True,
    # Timedelta is only editable if it is explicitly configured as number column.
    # Reason is that the timedelta is provided as nanoseconds as default, which
    # quickly overflows the max value supported by Number in Javascript.
    column_config={"timedelta": st.column_config.NumberColumn()},
)

st.subheader("Period dtypes in pd.DataFrame")
st.data_editor(PERIOD_TYPES_DF, use_container_width=True)

st.subheader("Unsupported types")
st.data_editor(UNSUPPORTED_TYPES_DF, use_container_width=True)
