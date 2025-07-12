import streamlit as st
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# Load the trained model
model = joblib.load('rf_model.pkl')

st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")

# Sidebar Navigation using option_menu
with st.sidebar:
    selected = option_menu(
            "Life Expectancy",             # App title in sidebar
            ["Model Info", "Predict"],            # Tabs
            icons=["bar-chart", "activity"],      # Optional icons
            menu_icon= " ",                 # Top icon
            default_index=0,                      # Default tab
            styles={
                "container": {"padding": "10px", "background-color": "#262730"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#656780"
                    },
                "nav-link-selected": {"background-color": "#57596E", "font-weight": "normal"},
                }
            )

# model info Page
if selected == "Model Info":
    st.title("Model Information & Performance")

    st.subheader("1. Project Overview")
    st.markdown("""
    This machine learning project predicts **life expectancy** based on various socio-economic and health-related factors.

    The dataset includes features such as adult mortality, immunization coverage, alcohol consumption, GDP, schooling years, and more.
    """)

    st.markdown("---")


    st.subheader("2. Dataset Details")
    st.markdown("""
    - **Source**: WHO Life Expectancy Dataset  
    - **Total Records**: ~2900 rows  
    - **Features Used**: 18 numeric features  
    - **Target Variable**: `Life expectancy` (in years)
    """)


    st.markdown("---")

    st.subheader("3. Model Used")
    st.markdown("""
    A **Random Forest Regressor** was used to predict life expectancy.

    - It combines multiple decision trees to improve prediction accuracy.
    - It's robust to overfitting and handles both linear and non-linear patterns.
    - Randomness is introduced by selecting random samples and random feature subsets for each tree.
    """)

    st.markdown("---")


    st.subheader("4. Model Settings")
    st.markdown("""
    - `n_estimators = 100`  
    - `max_depth = None`  
    - `min_samples_split = 2`  
    - `random_state = 42`
    """)

    st.markdown("---")

    st.subheader("5. Model Performance (Validation Set)")
    st.metric("Mean Absolute Error (MAE)", "1.17 years")
    st.metric("Mean Squared Error (MSE)", "3.15")
    st.metric("R² Score", "0.9658")

# prediction Page
elif selected == "Predict":
    st.title("Predict Life Expectancy")
st.write("### Enter feature values")

col1, col2 = st.columns(2)
with col1:
    adult_mortality = st.number_input("Adult Mortality", value=250.0)
with col2:
    infant_deaths = st.number_input("Infant Deaths", value=50)

col1, col2 = st.columns(2)
with col1:
    alcohol = st.number_input("Alcohol Consumption (litres)", value=3.5)
with col2:
    expenditure = st.number_input("Health Expenditure (%)", value=100.0)

col1, col2 = st.columns(2)
with col1:
    hep_b = st.number_input("Hepatitis B Coverage (%)", value=90)
with col2:
    measles = st.number_input("Measles Cases", value=200)

col1, col2 = st.columns(2)
with col1:
    bmi = st.number_input("BMI", value=20.0)
with col2:
    under5 = st.number_input("Under-5 Deaths", value=60)

col1, col2 = st.columns(2)
with col1:
    polio = st.number_input("Polio Coverage (%)", value=80)
with col2:
    total_exp = st.number_input("Total Health Expenditure (%)", value=7.0)

col1, col2 = st.columns(2)
with col1:
    diphtheria = st.number_input("Diphtheria Coverage (%)", value=85)
with col2:
    hiv_aids = st.number_input("HIV/AIDS Deaths (0–4)", value=0.1)

col1, col2 = st.columns(2)
with col1:
    gdp = st.number_input("GDP (USD)", value=1000.0)
with col2:
    population = st.number_input("Population", value=1_000_000.0)

col1, col2 = st.columns(2)
with col1:
    thin_19 = st.number_input("Thinness Age 10–19 (%)", value=5.0)
with col2:
    thin_5_9 = st.number_input("Thinness Age 5–9 (%)", value=5.0)

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Income Composition of Resources", value=0.6)
with col2:
    schooling = st.number_input("Years of Schooling", value=12.0)



if st.button("Predict"):
    input_data = np.array([[adult_mortality, infant_deaths, alcohol, expenditure,
                                hep_b, measles, bmi, under5, polio,
                                total_exp, diphtheria, hiv_aids, gdp, population,
                                thin_19, thin_5_9, income, schooling]])

    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Life Expectancy: {prediction:.2f} years")

