import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from scripts.load_data import LoadData
db = LoadData()
db.connect()
df = db.fetch_data("SELECT * FROM xdr_data;")


# Set page title
st.title("Telecom Data")

st.write(df.head())

st.write(df.describe())


# Option to drop or fill missing values
st.subheader("Handle Missing Values")
missing_column = st.selectbox("Choose a column to fill missing values", df.columns[df.isnull().any()])
fill_method = st.radio("Fill method", ["Fill with Mean", "Fill with Median", "Drop Rows"])

if st.button("Apply Fill"):
    if fill_method == "Fill with Mean":
        df[missing_column] = df[missing_column].fillna(df[missing_column].mean())
    elif fill_method == "Fill with Median":
        df[missing_column] = df[missing_column].fillna(df[missing_column].median())
    else:
        df = df.dropna(subset=[missing_column])
    st.success(f"{missing_column} cleaned successfully")

st.subheader("Descriptive Statistics")
st.write(df.describe())


handset_counts = df['Handset Type'].value_counts()
st.bar_chart(handset_counts)

df_clean = df.dropna(subset=["Handset Type", "Total DL (Bytes)"])  # Clean data

fig, ax = plt.subplots()
sns.boxplot(x="Handset Type", y="Handset Manufacturer", data=df_clean, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)


