import pandas as pd 
import numpy as np
import pickle
import streamlit as st
from PIL import Image 

df = pd.read_csv('loan_data.csv')
df.head()