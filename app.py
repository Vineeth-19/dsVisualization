import streamlit as st
from multiapp import MultiApp
from apps import home, data, model # import your app modules here

app = MultiApp()
app.add_app("Constituency", home.app)
app.add_app("Candidate Participation and Party Performance", data.app)
app.add_app("Candidate History and Personal Information", model.app)
# The main app
st.markdown('<h1 style="color: orange;">Lok Sabha Elections: Trends & Results</h1>', unsafe_allow_html=True)
app.run()



# Add all your application here

