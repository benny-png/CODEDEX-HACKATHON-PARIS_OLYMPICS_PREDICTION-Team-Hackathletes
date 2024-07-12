import streamlit as st
import pandas as pd
import joblib

# Load the trained models
with st.spinner("Loading models..."):
    swimming_model = joblib.load('swimming_time_taken_predictor.joblib')
    medals_model = joblib.load('country_medals_predictor.joblib')

# Assuming you have the accuracy scores
swimming_model_accuracy = 0.14337334783858943  # Replace with your actual accuracy score
medals_model_accuracy = 0.8055  # Replace with your actual accuracy score

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the prediction model", ["Athlete's Swimming Time", "Country's Winning Medals"])

# Dictionary to map full team names to their abbreviations
team_abbreviations = {
    'Algeria': 'ALG', 'Argentina': 'ARG', 'Australia': 'AUS', 'Belgium': 'BEL', 'Brazil': 'BRA',
    'Canada': 'CAN', 'Croatia': 'CRO', 'Cuba': 'CUB', 'Unified Team': 'EUN', 'France': 'FRA',
    'West Germany': 'FRG', 'Great Britain': 'GBR', 'East Germany': 'GDR', 'Germany': 'GER',
    'Hungary': 'HUN', 'Italy': 'ITA', 'Japan': 'JPN', 'South Korea': 'KOR', 'Netherlands': 'NED',
    'Puerto Rico': 'PUR', 'Russian Olympic Committee': 'ROC', 'Romania': 'ROU', 'South Africa': 'RSA',
    'Russia': 'RUS', 'Switzerland': 'SUI', 'Sweden': 'SWE', 'Ukraine': 'UKR', 'Soviet Union': 'URS',
    'United States': 'USA', 'Venezuela': 'VEN'
}

team_columns = [f'Team_{abbr}' for abbr in team_abbreviations.values()]

if app_mode == "Athlete's Swimming Time":
    st.title("Paris Olympics Swimming Predictions 🚀")
    st.write("Predict the winning times for freestyle swimming events at the Paris Olympics based on historical data.")
    st.write(f"Mean Squared Error (MSE): {swimming_model_accuracy}")

    # Input fields for swimming event prediction
    st.warning("Input Athletes Information (Model Features)")
    year = st.number_input("Year (for this case 2024)", min_value=1896, max_value=2024, step=1)
    relay = st.selectbox("Relay Event ?", [False, True])
    rank = st.number_input("Athlete's previous Rank", min_value=1, max_value=10, step=1)
    team = st.selectbox("Team ( Country )", list(team_abbreviations.keys()))

    # Prepare the input data for swimming model
    team_abbreviation = team_abbreviations[team]
    swimming_input_data = pd.DataFrame({
        'Year': [year],
        'Distance (in meters)': [100],  # Assuming 100 meters distance for simplicity
        'Relay?': [relay],
        'Rank': [rank],
        'Location_Amsterdam': [False],
        'Location_Angeles': [False],
        'Location_Athens': [False],
        'Location_Atlanta': [False],
        'Location_Barcelona': [False],
        'Location_Beijing': [False],
        'Location_Berlin': [False],
        'Location_City': [False],
        'Location_Helsinki': [False],
        'Location_London': [False],
        'Location_Melbourne': [False],
        'Location_Montreal': [False],
        'Location_Moscow': [False],
        'Location_Munich': [False],
        'Location_Paris': [True],
        'Location_Rio': [False],
        'Location_Rome': [False],
        'Location_Seoul': [False],
        'Location_Sydney': [False],
        'Location_Tokyo': [False],
        'Stroke_Freestyle': [True],  # Assuming Freestyle stroke for simplicity
        'Gender_Men': [True],  # Assuming Men gender for simplicity
    })

    for col in team_columns:
        swimming_input_data[col] = False
    swimming_input_data[f'Team_{team_abbreviation}'] = True

    # Make predictions for swimming event
    if st.button('Predict Swimming Time'):
        with st.spinner('Predicting...'):
            swimming_prediction = swimming_model.predict(swimming_input_data)

        # Display the prediction
        st.success(f"Predicted Winning Time: {swimming_prediction[0]:.2f} seconds")

    # Display the trend of average times per decade
    st.header("Trend of Average Times per Decade 📈")
    trend_image_path = 'trend_average_times_per_decade.png'
    st.image(trend_image_path, caption='Trend of Average Times per Decade', width=850)
    st.markdown("[View Detailed Analysis](https://colab.research.google.com/drive/1KrlXwtxwTIS7rlluiuiYtojrf7tAL6yP#scrollTo=OAzOQSAk-X_z)")

elif app_mode == "Country's Winning Medals":
    st.title("Paris Olympics Country's Medal Predictions 🏅")
    st.write("Predict the total number of medals a country will win at the Paris Olympics based on historical data.")
    st.write(f"Model Accuracy: {medals_model_accuracy * 100:.2f}%, Mean Absolute Error: 3.08")

    # Input fields for country medal prediction
    st.warning("Input Country's Information (Model Features)")
    athlete_count = st.number_input("Number of Athletes", min_value=1, step=1, value=300)
    coach_count = st.number_input("Number of Coaches", min_value=1, step=1, value=50)
    total_participants = athlete_count + coach_count
    gold = st.number_input("Historical Gold Medals", min_value=0, step=1, value=39)
    silver = st.number_input("Historical Silver Medals", min_value=0, step=1, value=41)
    bronze = st.number_input("Historical Bronze Medals", min_value=0, step=1, value=33)

    # Prepare the input data for medals model
    medals_input_data = pd.DataFrame({
        'Athlete_Count': [athlete_count],
        'Coach_Count': [coach_count],
        'Total_Participants': [total_participants],
        'Gold': [gold],   # Using historical data
        'Silver': [silver], # Using historical data
        'Bronze': [bronze]  # Using historical data
    })

    # Make predictions for country medals
    if st.button('Predict Country Medals'):
        with st.spinner('Predicting...'):
            medals_prediction = medals_model.predict(medals_input_data)

        # Display the prediction
        st.success(f'Predicted Total Medals: {int(round(medals_prediction[0]))}')

    # Display the chloropeth map of world medal distribution
    st.header("Tokyo Olympics World Medal Distribution")
    chloropeth_image_path = 'medals_chloropethmap.png'
    st.image(chloropeth_image_path, caption='World Medal Distribution', width=850)
    st.markdown("[View Detailed Analysis](https://colab.research.google.com/drive/16i3dut1C2pRn9Olcgd_mjv5RfqlNCl5Z#scrollTo=1mxp1upio6ad)")
