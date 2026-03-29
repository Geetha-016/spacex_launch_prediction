import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="SpaceX Predictor", layout="centered")

st.title("🚀 SpaceX Launch Success Prediction")
st.write("Predict whether Falcon 9 first stage will land successfully.")

# Load model and scaler
model = joblib.load("spacex_model/model.joblib")
scaler = joblib.load("spacex_model/scaler.pkl")

# -------- USER INPUTS --------
flight_number = st.number_input("Flight Number", 1, 200, 50)
payload = st.slider("Payload Mass (kg)", 0, 10000, 5000)
flights = st.number_input("Number of Flights", 1, 10, 1)
block = st.number_input("Block Version", 1, 10, 5)
reused_count = st.number_input("Reused Count", 0, 10, 0)

orbit = st.selectbox("Orbit Type", [
    "LEO", "GTO", "ISS", "SSO"
])

launch_site = st.selectbox("Launch Site", [
    "CCAFS SLC 40",
    "KSC LC 39A",
    "VAFB SLC 4E"
])

# -------- ALL FEATURES (83 columns) --------
all_columns = [
'FlightNumber','PayloadMass','Flights','Block','ReusedCount',
'Orbit_ES-L1','Orbit_GEO','Orbit_GTO','Orbit_HEO','Orbit_ISS',
'Orbit_LEO','Orbit_MEO','Orbit_PO','Orbit_SO','Orbit_SSO',
'Orbit_VLEO','LaunchSite_CCAFS SLC 40','LaunchSite_KSC LC 39A',
'LaunchSite_VAFB SLC 4E','LandingPad_5e9e3032383ecb267a34e7c7',
'LandingPad_5e9e3032383ecb554034e7c9','LandingPad_5e9e3032383ecb6bb234e7ca',
'LandingPad_5e9e3032383ecb761634e7cb','LandingPad_5e9e3033383ecbb9e534e7cc',
'Serial_B0003','Serial_B0005','Serial_B0007','Serial_B1003','Serial_B1004',
'Serial_B1005','Serial_B1006','Serial_B1007','Serial_B1008','Serial_B1010',
'Serial_B1011','Serial_B1012','Serial_B1013','Serial_B1015','Serial_B1016',
'Serial_B1017','Serial_B1018','Serial_B1019','Serial_B1020','Serial_B1021',
'Serial_B1022','Serial_B1023','Serial_B1025','Serial_B1026','Serial_B1028',
'Serial_B1029','Serial_B1030','Serial_B1031','Serial_B1032','Serial_B1034',
'Serial_B1035','Serial_B1036','Serial_B1037','Serial_B1038','Serial_B1039',
'Serial_B1040','Serial_B1041','Serial_B1042','Serial_B1043','Serial_B1044',
'Serial_B1045','Serial_B1046','Serial_B1047','Serial_B1048','Serial_B1049',
'Serial_B1050','Serial_B1051','Serial_B1054','Serial_B1056','Serial_B1058',
'Serial_B1059','Serial_B1060','Serial_B1062','GridFins_False','GridFins_True',
'Reused_False','Reused_True','Legs_False','Legs_True'
]

# -------- CREATE EMPTY DATAFRAME --------
df = pd.DataFrame([[0]*len(all_columns)], columns=all_columns)

# -------- FILL USER INPUTS --------
df['FlightNumber'] = flight_number
df['PayloadMass'] = payload
df['Flights'] = flights
df['Block'] = block
df['ReusedCount'] = reused_count

# Orbit mapping
orbit_column = f"Orbit_{orbit}"
if orbit_column in df.columns:
    df[orbit_column] = 1

# Launch site mapping
launch_column = f"LaunchSite_{launch_site}"
if launch_column in df.columns:
    df[launch_column] = 1

# -------- PREDICTION --------
if st.button("Predict Launch Outcome"):
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    if prediction == 1:
        st.success("✅ Successful Landing")
        st.write(f"Confidence: {prob[1]*100:.2f}%")
    else:
        st.error("❌ Landing Failed")
        st.write(f"Confidence: {prob[0]*100:.2f}%")

st.caption("This app predicts SpaceX Falcon 9 landing success using Machine Learning.")
