# SpaceX Falcon 9 First Stage Landing Prediction  
IBM Data Science Capstone Project  

[![Open Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://spacexlaunchprediction-tsp49vu9kxuv5ghgkctppc.streamlit.app/)

---

## Background
SpaceX, a leader in the space industry, strives to make space travel affordable for everyone. Its accomplishments include sending spacecraft to the International Space Station, launching a satellite constellation that provides internet access, and sending manned missions to space.

SpaceX reduces launch costs (approximately $62 million per launch) by reusing the first stage of the Falcon 9 rocket. Other providers, who do not reuse the first stage, can charge upwards of $165 million per launch.

By predicting whether the first stage will successfully land, this project helps estimate overall launch cost efficiency using public data and machine learning models.

---

## Explore
* How payload mass, launch site, number of flights, and orbit type affect first-stage landing success  
* Rate of successful landings over time  
* Identify the best predictive model for successful landing (binary classification)

---

## Executive Summary
This project identifies key factors influencing successful Falcon 9 first-stage landings.

The following methodologies were used:  
* Collect data using the SpaceX REST API and web scraping techniques  
* Wrangle data to create a success/fail outcome variable  
* Explore data using visualizations and SQL analysis  
* Build Models to predict landing outcomes using Logistic Regression, Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbor (KNN)  
* Deploy model using Streamlit for interactive predictions  

---

# Results

## Exploratory Data Analysis
* Launch success rate has improved over time  
* KSC LC-39A has the highest success rate among landing sites  
* Orbits ES-L1, GEO, HEO, and SSO show a 100% success rate

## Visualization / Analytics
* Explored relationships between payload, booster version, and landing outcomes  
* Identified trends in historical launch data that influence model performance

## Predictive Analytics
* All models performed similarly on the test dataset  
* The Decision Tree Classifier slightly outperformed other models based on .best_score_ from GridSearchCV

---

# Methodology

## Data Collection – API
* Request launch data from SpaceX API  
* Decode JSON response using .json()  
* Convert data into DataFrame using .json_normalize()  
* Filter dataset to include only Falcon 9 launches  
* Replace missing Payload Mass values with calculated mean  
* Export cleaned dataset to CSV

## Data Collection – Web Scraping
* Scrape Falcon 9 launch data from Wikipedia  
* Use BeautifulSoup to parse HTML tables  
* Extract relevant columns and convert to DataFrame  
* Export dataset to CSV

## Data Wrangling
* Convert landing outcomes into binary values:  
  * 1 → Successful landing  
  * 0 → Unsuccessful landing

## EDA with Visualization
* Created scatter plots, bar charts, and trend analysis visualizations

## EDA with SQL
* Queried database to calculate:  
  * Total payload mass  
  * Payload range for successful launches  
  * Total number of successful vs failed landings

## Maps with Folium
* Visualized launch sites  
* Mapped landing outcomes  
* Calculated distances to nearby geographical locations

## Dashboard with Plotly Dash
* Interactive dashboard including:
  * Pie chart for launch success by site  
  * Scatter plot showing Payload Mass vs Success Rate by Booster Version

## Deployment with Streamlit
* Deployed the trained predictive model using Streamlit  
* Interactive app allows users to input payload, booster version, and launch site to predict landing success  
* [Open the Streamlit App](https://spacexlaunchprediction-tsp49vu9kxuv5ghgkctppc.streamlit.app/)

## Predictive Analytics
* Created NumPy array from Class column  
* Standardized features using StandardScaler  
* Split dataset into training and test sets using train_test_split  
* Used GridSearchCV (cv=10) for hyperparameter tuning  
* Applied models:  
  * Logistic Regression  
  * Support Vector Machine (SVC)  
  * Decision Tree Classifier  
  * K-Nearest Neighbor Classifier  
* Evaluated models using:  
  * Accuracy  
  * Confusion Matrix   
  * Classification Report (includes Precision, Recall, F-1 Score)

---

# Conclusion
* Model Performance: Decision Tree Classifier slightly outperformed others  
* Equator Advantage: Launch sites near the equator benefit from Earth's rotational speed  
* Coastal Locations: All launch sites are close to the coast  
* Launch Success Trend: Overall success rate has increased over time  
* KSC LC-39A: Highest success rate; 100% success for payloads under 5,500 kg  
* Orbit Success: ES-L1, GEO, HEO, and SSO show 100% landing success  
* Payload Mass: Higher payload mass is associated with higher success rate

---

## Tools & Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, SQL, BeautifulSoup, Plotly, Folium, Scikit-learn, Streamlit
