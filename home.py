import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Election Data Analysis", page_icon=":bar_chart:", layout="wide")
st.markdown(
    """
    <style>
        .reportview-container {
            background: #f5f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
df = pd.read_csv('LS_2.0.csv')

# Clean and preprocess data
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace('\n', '_')
df.rename(columns={'over total electors _in constituency': 'total_voters',
                   'over total votes polled _in constituency': 'votes_polled',
                   'total electors': 'total_electors'}, inplace=True)

def value_cleaner(x):
    try:
        str_temp = (x.split('Rs')[1].split('\n')[0].strip())
        str_temp_2 = ''.join(str_temp.split(","))
        return str_temp_2
    except:
        return 0

df['assets'] = df['assets'].apply(value_cleaner)
df['liabilities'] = df['liabilities'].apply(value_cleaner)
df['education'] = df['education'].str.replace('\n', '')
df['party'] = df['party'].str.replace('TRS', 'BRS')
df = df.fillna(0)
df['criminal_cases'] = df['criminal_cases'].replace({'Not Available': 0})
df['criminal_cases'] = pd.to_numeric(df['criminal_cases'], errors='coerce').astype(np.int64)
df['age'] = df['age'].apply(lambda x: round(x))

# Convert columns to numeric
numerical_columns = ['assets', 'liabilities', 'age']
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

def app():
    st.sidebar.image('election.jpg', caption='Election Comission of India')
    
    
    

# Correlation Heatmap
    st.markdown('<h2 style="color:#cc91c9;">Heatmap</h2>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    fig_corr, ax_corr = plt.subplots(figsize=(10,5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    st.pyplot(fig_corr)
    state = df.groupby('state').apply(lambda x: x['constituency'].nunique()).reset_index(name='constituency')

# Load shapefile
    shp_gdf = gpd.read_file('Indian_States.shp')  # Replace 'Indian_States.shp' with the actual file name

# Merge data
    merged = shp_gdf.set_index('st_nm').join(state.set_index('state'))

    st.markdown('<h2 style="color:#cc91c9;">State-wise Distribution of Indian Constituencies</h2>', unsafe_allow_html=True)
    fig_state_distribution = px.bar(state, x='state', y='constituency',
                                color='constituency',
                                labels={'constituency': 'Constituencies'},
                                title='Statewise Distribution of Constituencies in India',
                                template='plotly_dark')
    st.plotly_chart(fig_state_distribution)

    state = df.groupby('state').apply(lambda x: x['constituency'].nunique()).reset_index(name='constituency')
    shp_gdf = gpd.read_file('Indian_States.shp')
    merged = shp_gdf.set_index('st_nm').join(state.set_index('state'))

# Streamlit app
    
    st.pyplot(plt, merged.plot(column='constituency', cmap='inferno_r', linewidth=0.5, edgecolor='0.2', legend=True))

#Sunburst Chart
    st.sidebar.title('Sunburst Chart Settings')

    visualization_option = st.sidebar.radio("Select Visualization", ["Entire Dataset", "Each State"])

    if visualization_option == "Entire Dataset":
    # Visualize for the entire dataset
        fig_sunburst = px.sunburst(df, path=['state', 'constituency'],
                               values='total_electors',
                               color='total_electors',
                               color_continuous_scale='viridis_r')
        fig_sunburst.update_layout(title_text='Sunburst Chart - Constituencies for All States',
                               template='plotly_dark')
        st.plotly_chart(fig_sunburst)

    else:
    # Visualize for each state
        selected_state = st.sidebar.selectbox('Select State', df['state'].unique())
        filtered_data = df[df['state'] == selected_state]

        fig_sunburst_state = px.sunburst(filtered_data, path=['state', 'constituency'],
                                     values='total_electors',
                                     color='total_electors',
                                     color_continuous_scale='viridis_r')
        fig_sunburst_state.update_layout(title_text=f'Sunburst Chart - {selected_state} Constituencies',
                                     template='plotly_dark')
        st.plotly_chart(fig_sunburst_state)

    
