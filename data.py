import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go



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
df1=df[df['party']!='NOTA']
# Assuming df1 is your DataFrame


# Streamlit app


vote_share_top5 = df1.groupby('party')['total_votes'].sum().nlargest(5).index.tolist()

def vote_share(row):
    if row['party'] not in vote_share_top5:
        return 'other'
    else:
        return row['party']


def app():
    st.sidebar.image('election.jpg', caption='Election Comission of India')
   # Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Gender Distribution in Participation</h2>', unsafe_allow_html=True)


    req = pd.DataFrame(df1['gender'])
# Create a Pie chart with Plotly Express and name it fig3
    fig3 = px.pie(req, names='gender')

# Update layout
    fig3.update_layout( template='plotly_dark')

# Display the Pie chart in Streamlit
    st.plotly_chart(fig3)
    gen = df1.groupby('gender').apply(lambda x: x['name'].count()).reset_index(name='counts')
    gen['category'] = 'Overall gender ratio'
    winners = df1[df1['winner'] == 1]
    gen_win = winners.groupby('gender').apply(lambda x: x['name'].count()).reset_index(name='counts')
    gen_win['category'] = 'Winning gender ratio'
    total = pd.concat([gen_win, gen])

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Participation vs Win Counts</h2>', unsafe_allow_html=True)

# Create a bar chart with Plotly Express
    fig4 = px.bar(total, x='gender', y='counts', color='category', barmode='group')
    fig4.update_layout(title_text='Participation vs Win Counts', template='plotly_dark')

# Display the bar chart in Streamlit
    st.plotly_chart(fig4)

    party_state = df1.groupby('party')['state'].nunique().reset_index(name='state')
    party_const = df1.groupby('party')['constituency'].count().reset_index(name='constituency')
    top_const = party_const.sort_values(by='constituency', ascending=False)[:25]
    top_party = pd.merge(top_const, party_state, how='inner', left_on='party', right_on='party')

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Constituency vs Statewise Participation for the Most Contesting Political Parties</h2>', unsafe_allow_html=True)
    fig5 = px.scatter(top_party, x='constituency', y='state', color='state', size='constituency', hover_data=['party'])
    fig5.update_layout(title_text='Constituency vs Statewise Participation for the Most Contesting Political Parties', template='plotly_dark')

# Display the scatter plot in Streamlit
    st.plotly_chart(fig5)
    st.markdown('<h2 style="color:#cc91c9;">Statewise Report Card for Political Parties in India</h2>', unsafe_allow_html=True)
    st.sidebar.title('Statewise Report Card Chart Settings')

# Checkbox for choosing to visualize all states
    all_states_option = st.sidebar.checkbox('Visualize All States', value=True)
    if all_states_option:
        filtered_data = winners
    else:
    # Selectbox to choose a specific state
        selected_state = st.sidebar.selectbox('Select a State', winners['state'].unique())
        filtered_data = winners[winners['state'] == selected_state]

    st_party = filtered_data.groupby(['party', 'state'])['winner'].sum().reset_index(name='wins')

# Pivot the table for the heatmap
    pivot_st_party = pd.pivot(st_party, index='party', columns='state', values='wins')

# Plot the heatmap with Seaborn
    plt.figure(figsize=(8, 6))  # Adjust the size as needed
    sns.heatmap(pivot_st_party, annot=True, fmt="g")

# Set labels and title
    plt.xlabel('State')
    plt.ylabel('Party')
    plt.title(f'Statewise Report Card for Political Parties', size=25)

# Display the heatmap in Streamlit
    st.pyplot(plt)
    df1['party_new'] = df1.apply(vote_share, axis=1)
    st.markdown('<h2 style="color:#cc91c9;">Vote Share Among Parties</h2>', unsafe_allow_html=True)
    st.sidebar.title('Total Vote Share Among Parties')
# Sidebar option to choose state or entire country
    visualization_option = st.sidebar.radio('Choose Party Wise Visualization Option', ['By State', 'Entire Country'])
    if visualization_option == 'By State':
    # Selectbox to choose a specific state
        selected_state = st.sidebar.selectbox('Staes', df1['state'].unique())
    
    # Filter data based on the selected state
        filtered_data = df1[df1['state'] == selected_state]

    # Group by party and calculate total votes
        counts = filtered_data.groupby('party_new')['total_votes'].sum().reset_index(name='votes')

    # Filter out 'other' category if it has zero votes
        counts = counts[counts['party_new'] != 'other']

        title = f'Partywise Vote Share in {selected_state}'
    else:
    # Visualize party share for the entire country
        counts = df1.groupby('party_new')['total_votes'].sum().reset_index(name='votes')

    # Filter out 'other' category if it has zero votes
        counts = counts[counts['party_new'] != 'other']

        title = 'Partywise Vote Share in Entire Country'

    fig6 = px.pie(counts, names='party_new', values='votes',
              title=title,
              hole=0.3,  # Set to 0 for a pie chart, non-zero for a doughnut chart
              labels={'party_new': 'Party'},
              color_discrete_sequence=px.colors.qualitative.Set3)

# Display the pie chart in Streamlit
    st.plotly_chart(fig6)