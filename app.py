import streamlit as st
import joblib
import pandas as pd

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="SpaceX Launch Predictor", layout="centered")
st.title("🚀 SpaceX Falcon 9 Landing Predictor")
st.write("Predict whether the first stage will land successfully based on launch parameters.")

# ----------------- LOAD MODEL & SCALER -----------------
model = joblib.load("spacex_model/model.joblib")
scaler = joblib.load("spacex_model/scaler.pkl")

# ----------------- USER INPUTS -----------------
st.sidebar.header("Enter Launch Parameters")

# Numeric inputs
flight_number = st.sidebar.number_input("Flight Number", 1, 200, 50)
payload = st.sidebar.slider("Payload Mass (kg)", 0, 10000, 5000)
flights = st.sidebar.number_input("Number of Flights", 1, 10, 1)
block = st.sidebar.number_input("Block Version", 1, 10, 5)
reused_count = st.sidebar.number_input("Reused Count", 0, 10, 0)

# Categorical inputs
orbit = st.sidebar.selectbox("Orbit Type", ["LEO", "GTO", "ISS", "SSO"])
launch_site = st.sidebar.selectbox("Launch Site", ["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"])

# Boolean features (Grid Fins, Reused, Legs)
grid_fins = st.sidebar.selectbox("Grid Fins", ["True", "False"])
reused = st.sidebar.selectbox("Reused", ["True", "False"])
legs = st.sidebar.selectbox("Legs", ["True", "False"])

# ----------------- FEATURE COLUMNS -----------------
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

# ----------------- CREATE DATAFRAME -----------------
df = pd.DataFrame([[0]*len(all_columns)], columns=all_columns)

# Fill numeric features
df.loc[0, 'FlightNumber'] = flight_number
df.loc[0, 'PayloadMass'] = payload
df.loc[0, 'Flights'] = flights
df.loc[0, 'Block'] = block
df.loc[0, 'ReusedCount'] = reused_count

# Fill categorical features
orbit_column = f"Orbit_{orbit}"
if orbit_column in df.columns:
    df.loc[0, orbit_column] = 1

launch_column = f"LaunchSite_{launch_site}"
if launch_column in df.columns:
    df.loc[0, launch_column] = 1

# Fill boolean features
df.loc[0, f"GridFins_{grid_fins}"] = 1
df.loc[0, f"Reused_{reused}"] = 1
df.loc[0, f"Legs_{legs}"] = 1

# ----------------- PREDICTION -----------------
if st.button("Predict Launch Outcome"):
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    if prediction == 1:
        st.success("✅ Landing Successful!")
        st.write(f"Confidence: {prob[1]*100:.2f}%")
    else:
        st.error("❌ Landing Failed")
        st.write(f"Confidence: {prob[0]*100:.2f}%")

# ----------------- DEBUG -----------------
# Uncomment to inspect inputs
# st.write(df)
