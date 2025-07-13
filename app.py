import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import plotly.express as px

# Load the trained model
model = joblib.load("rf_model.pkl")

st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")

# Sidebar Navigation using option_menu
with st.sidebar:  # Custom title with bigger font
    st.markdown(
        """
        <h2 style='color:white; font-size:24px; padding-left: 10px;'>Life Expectancy (RFR)</h2>
        """,
        unsafe_allow_html=True,
    )
    selected = option_menu(
        "",
        ["Model Info", "Predict", "Visualization", "Code Snippets", "About"],  # Tabs
        icons=["bar-chart", "activity", "display", "code-slash", "info"],  # Optional icons
        menu_icon=" ",  # top icon
        default_index=0,  # default tab
        styles={
            "container": {"padding": "10px", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#656780",
            },
            "nav-link-selected": {
                "background-color": "#57596E",
                "font-weight": "normal",
            },
        },
    )

# ------------------------------------------------------------------


# model info Page
if selected == "Model Info":
    st.title("Model Information & Performance")

    st.subheader("1. Project Overview")
    st.markdown(
        """
    This machine learning project predicts **life expectancy** based on various socio-economic and health-related factors.

    The dataset includes features such as adult mortality, immunization coverage, alcohol consumption, GDP, schooling years, and more.
    """
    )

    st.markdown("---")

    st.subheader("2. Dataset Details")
    st.markdown(
        """
    - **Source**: WHO Life Expectancy Dataset  
    - **Total Records**: ~2900 rows  
    - **Features Used**: 18 numeric features  
    - **Target Variable**: `Life expectancy` (in years)
    """
    )

    st.markdown("---")

    st.subheader("3. Model Used")
    st.markdown(
        """
    A **Random Forest Regressor** was used to predict life expectancy.

    - It combines multiple decision trees to improve prediction accuracy.
    - It's robust to overfitting and handles both linear and non-linear patterns.
    - Randomness is introduced by selecting random samples and random feature subsets for each tree.
    """
    )

    st.markdown("---")

    st.subheader("4. Model Settings")
    st.markdown(
        """
    - `n_estimators = 100`  
    - `max_depth = None`  
    - `min_samples_split = 2`  
    - `random_state = 42`
    """
    )

    st.markdown("---")

    st.subheader("5. Model Performance")
    st.metric("Mean Absolute Error (MAE)", "1.17 years")
    st.metric("Mean Squared Error (MSE)", "3.15")
    st.metric("R² Score", "0.9658")


# ------------------------------------------------------------------

# prediction Page
elif selected == "Predict":
    st.title("Predict Life Expectancy")
    st.write("### Enter only a few key values (others are pre-filled)")

    col1, col2 = st.columns(2)
    # ------------------------------------------------------------------
    with col1:
        hiv_aids = st.number_input("HIV/AIDS Deaths (0–4)", value=0.1)
        adult_mortality = st.number_input("Adult Mortality", value=250.0)
        under5 = st.number_input("Under-5 Deaths", value=60)
        alcohol = st.number_input("Alcohol Consumption (litres)", value=3.5)
    with col2:
        income = st.number_input("Income Composition of Resources (0–1)", value=0.6)
        schooling = st.number_input("Years of Schooling", value=12.0)
        bmi = st.number_input("BMI", value=20.0)
        thin_5_9 = st.number_input("Thinness Age 5–9 (%)", value=5.0)

    if st.button("Predict", use_container_width=True):
        # Fill in default values for remaining features
        default_values = {
            "infant_deaths": 30,
            "percentage_expenditure": 100.0,
            "Hepatitis B": 80,
            "Measles": 250,
            "Polio": 85,
            "Total expenditure": 7.0,
            "Diphtheria": 85,
            "GDP": 1000.0,
            "Population": 1_000_000.0,
            "thinness 1-19 years": 5.0,
        }

        # Full ordered input for prediction (18 features total)
        input_data = np.array(
            [
                [
                    adult_mortality,
                    default_values["infant_deaths"],
                    alcohol,
                    default_values["percentage_expenditure"],
                    default_values["Hepatitis B"],
                    default_values["Measles"],
                    bmi,
                    under5,
                    default_values["Polio"],
                    default_values["Total expenditure"],
                    default_values["Diphtheria"],
                    hiv_aids,
                    default_values["GDP"],
                    default_values["Population"],
                    thin_5_9,
                    default_values["thinness 1-19 years"],
                    income,
                    schooling,
                ]
            ]
        )

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Life Expectancy: {prediction:.2f} years")


# ------------------------------------------------------------------

# visualization page
if selected == "Visualization":
    st.title("Scatter Plot: Actual vs Predicted Life Expectancy")

    # Load saved test targets and predictions
    y_test = pd.read_csv("y_test.csv")
    y_pred = pd.read_csv("y_pred_test.csv")

    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame({
        "Actual": y_test.squeeze(),  # remove extra dimension if any
        "Predicted": y_pred.squeeze()
        })
    fig = px.scatter(
        df_plot,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Life Expectancy",
        labels={"Actual": "Actual Life Expectancy", "Predicted": "Predicted Life Expectancy"},
        trendline="ols",
        )

    fig.add_shape(
        type="line",
        x0=df_plot["Actual"].min(),
        y0=df_plot["Actual"].min(),
        x1=df_plot["Actual"].max(),
        y1=df_plot["Actual"].max(),
        line=dict(color="red", dash="dash"),
        )

    fig.update_layout(
        plot_bgcolor="#323C52",
        paper_bgcolor="#323C52"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.title("Feature Importance (Bar Chart)")

    # Load model and test features
    model = joblib.load("rf_model.pkl")
    X_test = pd.read_csv("X_test.csv")

    # Get feature importances
    importances = model.feature_importances_
    feature_names = X_test.columns

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Bar chart using Plotly
    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Random Forest)",
        color="Importance",
        color_continuous_scale="Blues",
    )

    fig2.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#323C52",
        paper_bgcolor="#323C52",
        font=dict(color="white"),
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.title("Correlation Heatmap (Top 5 Features + Life Expectancy)")

    # Load model, test features, and cleaned dataset
    model = joblib.load("rf_model.pkl")
    X_test = pd.read_csv("X_test.csv")
    df_cleaned = pd.read_csv("cleaned_life_expectancy_data.csv")

    # Get top 5 most important feature names
    importances = model.feature_importances_
    feature_names = X_test.columns
    top_indices = np.argsort(importances)[-5:]  # Top 5
    top_features = feature_names[top_indices].tolist()

    # Add target column to list
    selected_columns = top_features + ["Life expectancy"]

    # Compute correlation matrix for selected columns
    corr_small = df_cleaned[selected_columns].corr()

    # Create heatmap
    import plotly.figure_factory as ff

    fig3 = ff.create_annotated_heatmap(
        z=np.round(corr_small.values, 2),
        x=list(corr_small.columns),
        y=list(corr_small.columns),
        colorscale="YlGnBu",
        showscale=True,
        hoverinfo="z"
    )

    fig3.update_layout(
        title_text="Correlation Heatmap (top 5 features)",
        xaxis=dict(tickangle=45),
        plot_bgcolor="#323C52",
        paper_bgcolor="#323C52",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(fig3, use_container_width=True)



# ------------------------------------------------------------------

# code snippet page
elif selected == "Code Snippets":
    st.markdown("""### Libraries""")
    st.code(
        """
# importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
    """,
        language="python",
    )

    st.markdown("""### Loading the Dataset""")
    st.code(
        """
# loading the dataset
df = pd.read_csv("LifeExpectancyData.csv")""",
        language="python",
    )

    st.markdown("""### Data Cleaning""")
    st.code(
        """
df.columns = df.columns.str.strip()

# drop colums that don't contribute to prediction
columns_to_drop = ['Country', 'Year', 'Status']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# checking if there's any missing values
print(df.isnull().sum())

# droping rows where life expectancy is empty as cannot train with these
df = df[df['Life expectancy'].notnull()]

# fill other missing values with column means
df = df.fillna(df.mean(numeric_only=True))

# checking if that worked
print(df.isnull().sum())""",
        language="python",
    )

    st.markdown("""### Data Splitting""")
    st.code(
        """
# separating featues and target
X = df.drop('Life expectancy', axis = 1) # inputs
y = df['Life expectancy']                # target

# splitting data into training and temp (validation + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=90) # 0.8 for training, 0.2 for temp (80%, 20%)

X_vali, X_test, y_vali, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=90) # 0.5 of 0.2 for each validating and testing (10%)""",
        language="python",
    )

    st.markdown("""### Initializing and Training""")
    st.code(
        """
# initializing the model
model = RandomForestRegressor(
    n_estimators = 100,    # no.of trees (default 100)
    max_depth = None,        # max depth of each tree (default is None i.e. grow until pure)
    min_samples_split = 2, # minimum samples needed to split a node (default 2)
    random_state = 90      # for reproducibility
)

# training the model on the training set
model.fit(X_train, y_train)
#saving the model
joblib.dump(model, 'rf_model.pkl') # save the model
    """,
        language="python",
    )

    st.markdown("""### Evaluation""")
    st.code(
        """
# evaluating
y_pred = model.predict(X_vali)

# evaluation of the model using MAE, MSE, R squared
mae = mean_absolute_error(y_vali, y_pred)
mse = mean_squared_error(y_vali, y_pred)
r2 = r2_score(y_vali, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)""",
        language="python",
    )

    st.markdown("""### Testing""")
    st.code(
        """
# predicting on the test set
y_test_pred = model.predict(X_test)

# final performance
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print("MODEL PERFORMANCE:")
print("Test MAE:", mae_test)
print("Test MSE:", mse_test)
print("Test R² Score:", r2_test)
""",
        language="python",
    )



# ------------------------------------------------------------------


# about page

elif selected == "About":
    st.markdown(
        """
        ## About This Project

        This is a machine learning project built to predict life expectancy of  using a Random Forest Regressor (RFR). It predicts the **life expectancy of a country in a specific year**, based on real-world data provided by the **World Health Organization (WHO)** and **United Nations**.
        The model used is a **Random Forest Regressor**, trained to estimate the average life expectancy (in years) for the selected combination of features. The dataset includes data from over **180 countries** between the years **2000 to 2015**.
        It allows users to input health and socio-economic features and get an estimated life expectancy as output.
        The goal is to apply real-world data and ML techniques in a simple, user-friendly way.

        ---

        ## About Me

        I'm **Pawan**, a student of Computer Science with love for all things related to computers. I am nterested in coding, machine learning, designing and so on.  
        This project is part of my AI coursework.
        
        [GitHub](https://github.com/xyzeep) | [LinkedIn](https://linkedin.com/in/pawan0)
        """
    )
