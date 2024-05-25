import streamlit as st

# Main function
def main():
    # Set the background image 
    background_image = """
 <style>
    [data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1614850523060-8da1d56ae167?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bGlnaHQlMjBjb2xvdXJ8ZW58MHx8MHx8fDA%3D");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height=
    background-position: center;  
    background-repeat: no-repeat;
    }
    </style>
   
    """
    
    st.markdown(background_image, unsafe_allow_html=True)

    

    input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
    st.markdown(input_style, unsafe_allow_html=True)


    # Your Streamlit content goes here
    st.markdown('<h1 style="color:orange;">PREDICT CO2 EMISSION USING ML ALGO</h1>', unsafe_allow_html=True)
    st.markdown(""" 	
#### Welcome to the CO2 Emissions Prediction Dashboard.
#### :arrow_left: Use the sidebar to navigate to the different sections of the app.
    """)
    st.markdown('<span style="color:orange;font-size:40px;">Objective</span>', unsafe_allow_html=True)
    st.write("""The primary objective of the project is to develop a model that can accurately predict CO2 emissions based on different engine features of cars""")

if __name__ == "__main__":
    main()
