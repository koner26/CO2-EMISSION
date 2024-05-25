import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
df1=pd.read_csv("co2_emissions (1).csv")
df=df1.drop_duplicates()
Numerical_columns = ['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)', 'fuel_consumption_comb(mpg)', 'co2_emissions']
df_encode = df[Numerical_columns]
X = df_encode.drop('co2_emissions', axis=1) # Independent variable
y = df_encode['co2_emissions'] # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Standardize data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Train Random Forest model
def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    r2_train = model.score(X_train, y_train)
    rf_r2_score = cross_val_score(model, X_train, y_train, cv=10, scoring="r2").mean()
    return train_rmse, test_rmse, r2_train, rf_r2_score

def main():

    # Set the background image 
    background_image = """
 <style>
    [data-testid="stAppViewContainer"] > .main {
    background-image: url("data:image/jpeg;base64,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");
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
    


    menu = ["CO2 EMISSON", "EDA"]
    choice = st.selectbox(label = "Menu", options = menu, index = 0)
    if  choice == "CO2 EMISSON":
        st.title ( 'predict CO2 Emission From Your Car')
        st.subheader('Vehicle Details')
        engine_size = st.number_input('Engine Size (Liters)', min_value=1.0, max_value=6.0, step=0.1, value=2.0)
        fuel_consumption_city = st.number_input('Fuel Consumption City (L/100km)', min_value=5.0, max_value=30.0, step=0.1, value=10.0)
        fuel_consumption_hwy = st.number_input('Fuel Consumption Highway (L/100km)', min_value=5.0, max_value=30.0, step=0.1, value=7.0)
        fuel_consumption_comb = st.number_input('Fuel Consumption Combined (L/100km)', min_value=5.0, max_value=30.0, step=0.1, value=8.0)
        fuel_consumption_mpg = st.number_input('Fuel Consumption Combined (mpg)', min_value=10, max_value=50, step=1, value=25)
        cylinders = st.selectbox("Select No. of cylinders", df['cylinders'].unique())
        st.markdown(
    """
    <style>
    /* Target the label element of the selectbox */
    .stSelectbox > label {
        color: red;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        st.markdown(
    """
    <style>
    .stNumberInput label {
        font-size: 70px !important;
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
    
        
        model = train_random_forest_model(X_train_std, y_train)
        if st.button('Predict'):
            user_input = np.array([[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]])
            prediction = model.predict(user_input)
            st.subheader('Predicted CO2 Emissions:')
            st.write(f'{prediction[0]:.2f} grams per kilometer')
    elif choice == 'EDA':
        st.title(body = "Exploratory Data Analysis :chart:")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("Data We collected from the source")
        st.subheader('Brands of Cars')
        df_brand = df['make'].value_counts().reset_index().rename(columns={'count':'Count'})
        plt.figure(figsize=(15, 6))
        fig1 = sns.barplot(data=df_brand, x="make", y="Count")
        plt.xticks(rotation=75)
        plt.title("All Car Companies and their Cars")
        plt.xlabel("Companies")
        plt.ylabel("Cars")
        plt.bar_label(fig1.containers[0], fontsize=7)
        st.pyplot()
        st.write(df_brand)
        #NEXT PLOT
        st.subheader('VEHICLE CLASS')
        df_vehicle=df1['vehicle_class'].value_counts().reset_index().rename(columns={'count':'Count'})
        plt.figure(figsize=(10, 6))
        fig2=sns.barplot(x='vehicle_class',y='Count', data=df_vehicle)
        plt.xticks(rotation=90)
        plt.xlabel('ALL CLASS OF VEHICLE IN DATA')
        plt.ylabel('NO.OF CARS PRESENT  (COUNT)')
        st.pyplot()
        st.write(df_vehicle)
        #NEXT PLOT
        st.subheader("Scatter plot")
        fig_scatter = px.scatter(
            data_frame=df,
            x="fuel_consumption_city",
            y="co2_emissions",
            color="fuel_type",
            opacity=0.5
        )
        # Display the scatter plot in Streamlit
        st.plotly_chart(fig_scatter)


        #NEXT PLOT
        # Pie Chart
        st.subheader("Pie plot")
        fuel_type_count = df.groupby('fuel_type').size().reset_index(name='count')
        fig_pie = px.pie(data_frame=fuel_type_count,
                 names="fuel_type",
                 values="count",
                 title="Fuel Type Distribution")
        fig_pie.update_layout(
            width=400,  # Adjust width as needed
            height=400  # Adjust height as needed
        )
        st.plotly_chart(figure_or_data=fig_pie, use_container_width=True)
        df_fuel=df1['fuel_type'].value_counts().reset_index().rename(columns={'count':'Count'})
        st.write(df_fuel)


        # Violin Plot
        st.subheader("Violin plot")
        fig_violin = px.violin(data_frame = df,
                                x          = "fuel_consumption_city",
                                color      = "fuel_type", )
        st.plotly_chart(figure_or_data = fig_violin, use_container_width = True)
if __name__ == "__main__":
    main()