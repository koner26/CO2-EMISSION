import streamlit as st
from home import main as home_main
from co2_emission import main as co2_emission_main

st.set_page_config(page_title="CO2 Emissions Dashboard", page_icon=":earth_asia:", layout="centered", initial_sidebar_state="auto")

PAGES = {
    "Home": home_main,
    "CO2 Emission": co2_emission_main
}

def main():
    st.sidebar.title("MENU")
    choice = st.sidebar.selectbox("Choose Below", list(PAGES.keys()))
    page = PAGES[choice]
    page()

if __name__ == "__main__":
    main()
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)