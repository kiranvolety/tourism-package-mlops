
import joblib        st.warning(f"Unlikely to purchase the Wellness Tourism Package (probability: {probability:.2%})")

import pandas as pd    else:

import streamlit as st        st.success(f"Likely to purchase the Wellness Tourism Package (probability: {probability:.2%})")

    if prediction == 1:

MODEL_PATH = "best_model.joblib"

    probability = model.predict_proba(input_df)[0][1]

model = joblib.load(MODEL_PATH)    prediction = model.predict(input_df)[0]



st.title("Wellness Tourism Package Purchase Prediction")    }])

st.write("Enter customer details to predict whether they are likely to purchase the Wellness Tourism Package.")        "MonthlyIncome": monthly_income,

        "Designation": designation,

age = st.number_input("Age", min_value=18, max_value=100, value=35)        "NumberOfChildrenVisiting": number_of_children_visiting,

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])        "OwnCar": own_car,

city_tier = st.selectbox("City Tier", [1, 2, 3])        "PitchSatisfactionScore": pitch_satisfaction_score,

duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)        "Passport": passport,

occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])        "NumberOfTrips": number_of_trips,

gender = st.selectbox("Gender", ["Male", "Female"])        "MaritalStatus": marital_status,

number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)        "PreferredPropertyStar": preferred_property_star,

number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)        "ProductPitched": product_pitched,

product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])        "NumberOfFollowups": number_of_followups,

preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])        "NumberOfPersonVisiting": number_of_person_visiting,

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])        "Gender": gender,

number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)        "Occupation": occupation,

passport = st.selectbox("Passport", [0, 1])        "DurationOfPitch": duration_of_pitch,

pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])        "CityTier": city_tier,

own_car = st.selectbox("Own Car", [0, 1])        "TypeofContact": type_of_contact,

number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)        "Age": age,

designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])    input_df = pd.DataFrame([{

monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)if st.button("Predict"):
