# Streamlit Core
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Libraries
import os
import sys
import gdown

dirloc = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirloc, 'Customer Behaviour'))

from customer_behaviour import customer_behaviour

st.query_params["Page"] = 'Customer Behaviour'

if st.query_params["Page"] == 'Customer Behaviour':
    customer_behaviour()

