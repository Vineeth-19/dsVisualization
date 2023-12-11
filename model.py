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
req = pd.DataFrame(df1['gender'])

# Streamlit app


vote_share_top5 = df1.groupby('party')['total_votes'].sum().nlargest(5).index.tolist()

def vote_share(row):
    if row['party'] not in vote_share_top5:
        return 'other'
    else:
        return row['party']

gen = df1.groupby('gender').apply(lambda x: x['name'].count()).reset_index(name='counts')
gen['category'] = 'Overall gender ratio'
winners = df1[df1['winner'] == 1]
gen_win = winners.groupby('gender').apply(lambda x: x['name'].count()).reset_index(name='counts')
gen_win['category'] = 'Winning gender ratio'
total = pd.concat([gen_win, gen])

def app():
    st.sidebar.image('election.jpg', caption='Election Comission of India')
    

    winners1 = winners[winners['criminal_cases'] != 0]
    winners0 = winners[winners['criminal_cases'] == 0]

    winners_cri = winners1.groupby('party')['name'].count().reset_index(name='candidates')
    winners_no_cri = winners0.groupby('party')['name'].count().reset_index(name='candidates')

    winners_cri['status'] = 'Pending Case'
    winners_no_cri['status'] = 'No Pending Case'

    final_winners = pd.concat([winners_no_cri, winners_cri])

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Winners with Criminal Cases vs No Criminal Cases in Parties</h2>', unsafe_allow_html=True)
    

# Sidebar option to choose visualization scope
    

    # Visualize for the entire dataset
    fig7 = px.bar(final_winners, x='party', y='candidates', color='status',
                  title='Winners with Criminal Cases vs No Criminal Cases in All Parties',
                  template='plotly_dark')

# Display the bar chart in Streamlit
    st.plotly_chart(fig7)

    age_cnt = df1.groupby(['age', 'gender'])['name'].count().reset_index(name='counts')

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Age Counts Distribution among the Candidates</h2>', unsafe_allow_html=True)
    st.sidebar.title('Age Counts Distribution among the Candidates')

# Create a sidebar for user input
    selected_genders = st.sidebar.multiselect('Select Genders', age_cnt['gender'].unique(), default=age_cnt['gender'].unique())

# Filter data based on the selected genders
    filtered_data = age_cnt[age_cnt['gender'].isin(selected_genders)]

# Create an interactive histogram with Plotly Express
    fig9 = px.histogram(filtered_data, x='age', y='counts', color='gender',
                    marginal='violin',
                    title='Age Counts Distribution among the Candidates',
                    template='plotly_dark')

# Display the interactive histogram in Streamlit
    st.plotly_chart(fig9)
    party_state = df1.groupby('party')['state'].nunique().reset_index(name='state')
    party_const = df1.groupby('party')['constituency'].count().reset_index(name='constituency')
    top_const = party_const.sort_values(by='constituency', ascending=False)[:25]
    top_party = pd.merge(top_const, party_state, how='inner', left_on='party', right_on='party')
    pt_avg_age = df1.groupby('party')['age'].mean().round().reset_index(name='avg_age')
    final_avg_age = pd.merge(top_party['party'], pt_avg_age, how='inner', left_on='party', right_on='party')
    final_avg_age = final_avg_age.sort_values(by='avg_age', ascending=False)

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Average Age of Candidates in Each Party</h2>', unsafe_allow_html=True)
    st.sidebar.title('Average Age of Candidates in Each Party')
# Create a sidebar for user input
    party_selection = st.sidebar.multiselect('Select Parties', final_avg_age['party'].unique(), default=final_avg_age['party'].unique())

# Filter data based on selected parties
    filtered_data = final_avg_age[final_avg_age['party'].isin(party_selection)]

# Create an interactive bar chart with Plotly Express
    fig10 = px.bar(filtered_data, x='party', y='avg_age', color='avg_age',
               title='Average Age of Candidates in Each Party',
               template='plotly_dark')

# Display the interactive bar chart in Streamlit
    st.plotly_chart(fig10)  
    cri_cases = df1.groupby('criminal_cases')['name'].count().reset_index(name='counts')

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Criminal Cases Counts Distribution among the Politicians</h2>', unsafe_allow_html=True)

# Create an interactive histogram with Plotly Express
    fig11 = px.histogram(cri_cases, x='criminal_cases', y='counts', marginal='violin',
                     title='Criminal Cases Counts Distribution among the Politicians',
                     template='plotly_dark')

# Display the interactive histogram in Streamlit
    st.plotly_chart(fig11)
    cat_overall = df1.groupby('category')['name'].count().reset_index(name='counts')
    cat_overall['status'] = 'Overall Category Counts'
    cat_win = winners.groupby('category')['name'].count().reset_index(name='counts')
    cat_win['status'] = 'Winner Category Counts'
    cat_overl_win = pd.concat([cat_win, cat_overall])

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Participation vs Winning among Categories</h2>', unsafe_allow_html=True)

# Create an interactive grouped bar chart with Plotly Express
    fig12 = px.bar(cat_overl_win, x='category', y='counts', color='status', barmode='group',
               title='Participation vs Winning among Categories', template='plotly_dark')

# Display the interactive grouped bar chart in Streamlit
    st.plotly_chart(fig12)
    ed_cnt = df1.groupby('education')['name'].count().reset_index(name='counts')
    ed_win_cnt = winners.groupby('education').apply(lambda x: x['party'].count()).reset_index(name='counts')

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Education Qualification Analysis</h2>', unsafe_allow_html=True)
    st.sidebar.title('Education Qualification Analysis')

# Sidebar option to choose overall or individual education
    analysis_option = st.sidebar.radio("Select Analysis", ["Overall", "Individual Education"])
    if analysis_option == "Overall":
    # Overall education qualification distribution
        fig = go.Figure(data=[go.Pie(labels=ed_cnt['education'], values=ed_cnt['counts'],
                                 pull=[0.1, 0.2, 0, 0.1, 0.2, 0, 0.1, 0.1, 0.2, 0, 0.1, 0.2],
                                 title='Education Qualification of all Candidates')])
        fig.update_layout(title_text='Overall Education Qualification of all Candidates',
                      template='plotly_dark')
        st.plotly_chart(fig)
    else:
    # Education qualification distribution of the winners
        selected_education = st.sidebar.selectbox("Select Education", ed_win_cnt['education'])
        filtered_winners = winners[winners['education'] == selected_education]

        fig2 = go.Figure(data=[go.Pie(labels=filtered_winners['party'], values=ed_cnt['counts'],
                                  title=f'Education Qualification: {selected_education} - Winners')])
        fig2.update_layout(title_text=f'Education Qualification: {selected_education} - Winners',
                       template='plotly_dark')
        try:
            st.plotly_chart(fig2)
        except KeyError:
            st.warning("No data available for the selected education qualification.")
    win_as_liab = winners.sort_values(by='assets', ascending=False)

# Streamlit app
    st.markdown('<h2 style="color:#cc91c9;">Assets vs Liabilities Analysis</h2>', unsafe_allow_html=True)
    st.sidebar.title('Assets vs Liabilities Analysis')

# Sidebar option to choose between each state or the entire dataset
    visualization_option = st.sidebar.radio("Select Visualization", ["Country", "Each State"])
    if visualization_option == "Entire Dataset":
    # Visualize for the entire dataset
        fig = px.scatter(win_as_liab, x='assets', y='liabilities', color='state',
                     size='assets', hover_data=(['name', 'party', 'constituency', 'state', 'winner']),
                     title='Assets vs Liabilities for the Winning Politicians')
        fig.update_layout(title_text='Assets vs Liabilities for the Winning Politicians', template='plotly_dark')
        st.plotly_chart(fig)
    else:
    # Visualize for each state
        selected_state = st.sidebar.selectbox("Select State", win_as_liab['state'].unique())
        filtered_data = win_as_liab[win_as_liab['state'] == selected_state]

        fig_state = px.scatter(filtered_data, x='assets', y='liabilities', size='assets',
                           hover_data=(['name', 'party', 'constituency', 'state', 'winner']),
                           title=f'Assets vs Liabilities for Winning Politicians in {selected_state}')
        fig_state.update_layout(title_text=f'Assets vs Liabilities for Winning Politicians in {selected_state}',
                            template='plotly_dark')
        st.plotly_chart(fig_state)