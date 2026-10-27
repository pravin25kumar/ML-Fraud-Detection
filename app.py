import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
@st.cache_data
def load_data():
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(100)),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'size': np.random.randint(10, 100, 100)
    })
    return df
df = load_data()
st.header('Data Visualization Examples')
# Built-in Streamlit charts
st.subheader('Built-in Charts')
st.line_chart(df.set_index('date')['value'])
st.area_chart(df.set_index('date')['value'])
st.bar_chart(df.groupby('category')['value'].mean())
# Matplotlib
st.subheader('Matplotlib')
fig, ax = plt.subplots()
ax.scatter(df['value'], df['size'], c=df['category'].astype('category').cat.codes)
ax.set_xlabel('Value')
ax.set_ylabel('Size')
ax.set_title('Scatter Plot')
st.pyplot(fig)
# Seaborn
st.subheader('Seaborn')
fig, ax = plt.subplots()
sns.boxplot(data=df, x='category', y='value', ax=ax)
st.pyplot(fig)
# Maps
st.subheader('Maps')
map_data = pd.DataFrame({
    'lat': [37.7749, 40.7128, 41.8781, 29.7604],
    'lon': [-122.4194, -74.0060, -87.6298, -95.3698],
    'size': [100, 200, 150, 80]
})
st.map(map_data)