# streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.features import GeoJsonTooltip 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def set_background():
    image_url = "https://images.pexels.com/photos/1334312/pexels-photo-1334312.jpeg?cs=srgb&dl=pexels-designstrive-1334312.jpg&fm=jpg"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        body, .stApp, .stText, .stTitle, .stMarkdown, .stSubheader, 
        .stButton>button, .stCheckbox label, .stSelectbox label,
        .stSlider label, .css-1j9du1m, .css-1l8yf9b, .css-18e3th9, .css-1d391kg, 
        .css-1av6r1p, h1, h2, h3, h4, h5, h6, label {{
            color: white !important;
        }}


        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stMultiSelect select {{
            color: white !important;
            background-color: rgba(0, 0, 0, 0.5) !important; /* Dark background */
        }}

        .stSidebar {{
            color: white !important;
            background-color: rgba(0, 0, 0, 0.7) !important;
        }}

        .stButton>button {{
            color: white !important;
            background-color: rgba(0, 0, 0, 0.6) !important;
            border-radius: 8px;
        }}

        .stSlider {{
            color: white !important;
        }}

        .css-1aumxhk, .css-1t8x7v1 {{
            background-color: transparent !important;
        }}

        </style>
        """, 
        unsafe_allow_html=True
    )


def crop_price_analysis():
    data = pd.read_excel("cropsPrices.xlsx")

    data.columns = data.columns.str.strip()

    data.replace("NT", np.nan, inplace=True)
    data.fillna(data.mean(numeric_only=True), inplace=True)

    data.set_index("District", inplace=True)
    st.title("Crop Prices Analysis Across Districts")

    st.subheader("Correlation Matrix")
    correlation = data.corr()
    st.write(correlation)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data, cmap="YlGnBu", annot=False)
    ax.set_title("Correlation Heatmap of Crop Prices")
    st.pyplot(fig)

    crop = st.selectbox("Select Crop", options=data.columns)

    if crop:
        st.subheader(f"{crop} Prices Across Districts")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=data.index, y=data[crop],ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel("District")
        ax.set_ylabel("Price")
        ax.set_title(f"{crop} Prices Across Districts")
        st.pyplot(fig)

def yearlyRainfall():
    data = pd.read_csv("yearlyIndianRainfall.csv")
    data.columns = data.columns.str.strip()
    rainfall_types = {
        "SW": ["Actual_SW", "Normal_SW"],
        "NE": ["Actual_NE", "Normal_NE"],
        "Winter": ["Actual_Win", "Normal_Win"],
        "Summer": ["Actual_Hot", "Normal_Hot"],
        "Whole": ["Actual_whole", "Normal_whole"]
    }
    st.title("Yearly Indian Rainfall Analysis")
    st.subheader("Yearly Rainfall Data")
    st.write(data)
    st.subheader("Yearly Rainfall Comparison")
    rain_type = st.selectbox("Select Rainfall Type", options=["SW", "NE", "Winter", "Summer", "Whole"])
    if rain_type in rainfall_types:
        actual_column, normal_column = rainfall_types[rain_type]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data, x="Year", y=actual_column, label=f"Actual {rain_type}", ax=ax)
        sns.lineplot(data=data, x="Year", y=normal_column, label=f"Normal {rain_type}", ax=ax)
        
        ax.set_title(f"Comparison of Actual and Normal {rain_type} Rainfall")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rainfall (mm)")
        ax.legend()
        
        st.pyplot(fig)

def districtRainfall():
    data = pd.read_excel("districtRain.xlsx")
    tn_map = gpd.read_file("tamilnadu_districts.geojson")

    tn_map["District"] = tn_map["dtname"].str.strip().str.title()
    data["District"] = data["District"].str.strip().str.title()
    merged = tn_map.merge(data, on="District", how="left")
    m = folium.Map(location=[10.8, 78.6], zoom_start=7, scrollWheelZoom=False)

    folium.Choropleth(
        geo_data=merged,
        name="Rainfall",
        data=merged,
        columns=["District", "Actual"], 
        key_on="feature.properties.District", 
        fill_color="YlGnBu",
        edgecolor='black',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="District Rainfall (mm)"
    ).add_to(m)

    folium.GeoJson(
        merged,
        name="District Names",
        tooltip=folium.GeoJsonTooltip(
            fields=["District"],
            aliases=["District: "],
            style="background-color: white; color: black; font-weight: bold; font-size: 12px;", 
        ),
        style_function=lambda feature: {
            "color": "black",  
            "weight": 1.5,
            "fillOpacity": 0.5,
        },
    ).add_to(m)
    folium_static(m)

def peakSowingAndHarvestingSeasons():
    data = pd.read_excel("sowingAndHarvesting.xlsx")
    data.columns = data.columns.str.replace("\n", "").str.strip()
    st.title("Peak Sowing and Harvesting Seasons")
    cities = data["District"].unique()
    crops = [
        "KURUVAI", "SAMBA", "NAVARAI", "CHOLAM", "CUMBU", "RAGI", "MAIZE", "SUGARCANE", 
        "GROUNDNUT", "GINGELLY"
    ]

    selected_city = st.selectbox("Select District", cities)
    selected_crop = st.selectbox("Select Crop", crops)
    submit_button = st.button("Submit")

    if submit_button:
        filtered_data = data[data["District"] == selected_city]
        st.write(f"Filtered data for {selected_city}:")
        st.write(filtered_data)
        crop_column_map = {
            "KURUVAI": ["KURUVAI Peak Sowing Season", "KURUVAI Peak Harvesting Season"],
            "SAMBA": ["SAMBA Peak Sowing Season", "SAMBA Peak Harvesting Season"],
            "NAVARAI": ["NAVARAI Peak Sowing Season", "NAVARAI Peak Harvesting Season"],
            "CHOLAM": ["CHOLAM Peak Sowing Season", "CHOLAM Peak Harvesting Season"],
            "CUMBU": ["CUMBU Peak Sowing Season", "CUMBU Peak Harvesting Season"],
            "RAGI": ["RAGI Peak Sowing Season", "RAGI Peak Harvesting Season"],
            "MAIZE": ["MAIZE Peak Sowing Season", "MAIZE Peak Harvesting Season"],
            "SUGARCANE": ["SUGARCANE Peak Sowing Season", "SUGARCANE Peak Harvesting Season"],
            "Groundnut": ["GROUNDNUT Peak Sowing Season", "GROUNDNUT Peak Harvesting Season"],
            "Gingelly": ["GINGELLY Peak Sowing Season", "GINGELLY Peak Harvesting Season"]
        }

        selected_columns = crop_column_map.get(selected_crop, [])
        if selected_columns:
            sowing_season_column = selected_columns[0]
            harvesting_season_column = selected_columns[1]

            if sowing_season_column in filtered_data.columns:
                sowing_season = filtered_data[sowing_season_column].iloc[0] 

            if harvesting_season_column in filtered_data.columns:
                harvesting_season = filtered_data[harvesting_season_column].iloc[0]  

            if sowing_season in ['--', '-', None] or pd.isna(sowing_season):
                st.markdown(f"<h3 style='font-size:30px;'>No best season for sowing {selected_crop} in {selected_city}.</h2>", unsafe_allow_html=True)
            else:
                sowing_season_months = sowing_season.split('-')
                if len(sowing_season_months) == 2:
                    start, end = sowing_season_months
                    st.markdown(f"<h3 style='font-size:30px;'>The best sowing season for {selected_crop} in {selected_city} is from {start} to {end}.</h2>", unsafe_allow_html=True)
                else:
                    start = sowing_season_months    
                    st.markdown(f"<h3 style='font-size:30px;'>The best sowing season for {selected_crop} in {selected_city} is  {sowing_season}.</h3>", unsafe_allow_html=True)
                

            if harvesting_season in ['--', '-', None] or pd.isna(harvesting_season):
                st.markdown(f"<h3 style='font-size:30px;'>No best season for harvesting {selected_crop} in {selected_city}.</h2>", unsafe_allow_html=True)
            else:
                harvesting_season_months = harvesting_season.split('-')
                if len(harvesting_season_months) == 2:
                    start, end = harvesting_season_months
                    st.markdown(f"<h3 style='font-size:30px;'>The best harvesting season for {selected_crop} in {selected_city} is from {start} to {end}.</h3>", unsafe_allow_html=True)
                else:
                    start = harvesting_season_months
                    st.markdown(f"<h3 style='font-size:30px;'>The best harvesting season for {selected_crop} in {selected_city} is {harvesting_season}.</h3>", unsafe_allow_html=True)

        else:
            st.write("No data available for the selected crop and city combination.")

def modelPred():
    st.title("CROP PRICE - Model Comparison using RMSE")
    df = pd.read_excel("cropsPrices.xlsx")
    rain = pd.read_excel("districtRain.xlsx")
    crop_production = pd.read_excel("cropProductioninTons.xlsx")
    extra=pd.read_excel("extra.xlsx")
    
    districts = df["District"]
    
    df['District'] = df['District'].str.lower()
    rain['District'] = rain['District'].str.lower()
    crop_production['District'] = crop_production['District'].str.lower()
    extra['District']=extra['District'].str.lower()

    rain = rain[['District', 'Actual']]  
    df = pd.merge(df, rain, on='District', how='left')  
    df = pd.merge(df, crop_production, on='District', how='left')  
    df = pd.merge(df,extra,on='District',how='left')
    df.replace("NT", np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    if "District" in df.columns:
        df.drop(columns=["District"], inplace=True)

    st.write("### Data Preview")
    st.write(df.head())

    filtered_columns = [col for col in df.columns if '_x' in col]

    target_col = st.selectbox("Select Target Variable", filtered_columns)
    production_col = target_col.replace("_x", "_y")
    feature_cols = ["Actual", production_col,"GroundWater",'Fertilizer'] 

    st.write("### Selected Features")
    st.write(feature_cols)

    if target_col and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            "SVR": SVR(kernel='linear', C=1.0)  # Use a simpler kernel
        }

        rmse_scores = {}
        r2_scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            rmse_scores[name] = rmse
            r2_scores[name] = r2

            st.write(f"### {name} Predictions")
            st.write(y_pred)

        st.write("### RMSE Scores")
        st.write(rmse_scores)

        st.write("### R² Scores")
        st.write(r2_scores)

        best_model_name = min(rmse_scores, key=rmse_scores.get)
        best_model = models[best_model_name]

        st.write("") 
        st.write(f"### Best Model: {best_model_name}")
        st.write(f"RMSE: {rmse_scores[best_model_name]:.4f} and R²: {r2_scores[best_model_name]:.4f}")
        st.write("") 

        X_scaled = scaler.transform(X) 
        y_pred_all = best_model.predict(X_scaled)

        results = pd.DataFrame({
            "District": districts,
            "Actual Price": y,
            "Predicted Price": y_pred_all
        })

        st.write("### Predicted Prices for Selected Crop")
        st.write(results)

        plt.figure(figsize=(10, 6))
        plt.plot(districts, y_pred_all, color='blue', alpha=0.7, label="Predicted Prices", marker="o")
        plt.plot(districts, y, color='black', alpha=0.7, label="Actual Prices", marker="s")
        plt.xticks(rotation=90)
        plt.title("Actual vs Predicted Prices by District")
        plt.xlabel("Districts")
        plt.ylabel("Prices")
        plt.legend()
        st.pyplot(plt)

        residuals = y - y_pred_all
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred_all, residuals, color="purple", alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.title("Residual Plot: Prediction Errors")
        plt.xlabel("Predicted Prices")
        plt.ylabel("Residuals (Actual - Predicted)")
        st.pyplot(plt)



def main():
    set_background()
    st.title("Tamil Nadu Agriculture 2023-24")
    st.markdown("<h3>Welcome to the Agriculture Data Visualization Platform!</h3>",unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        /* Make all radio button text white */
        div[data-testid="stRadio"] label p {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white;'>Select an option</h4>", unsafe_allow_html=True)

    option = st.radio(
        "",
        options=[
            "Home", 
            "Analysis of Crop Prices Across Districts", 
            "Yearly Rainfall",
            "District wise Rainfall",
            "Peak Sowing and Harvesting Seasons",
            "Model prediction"
        ]
        )
    if option == "Home":
        st.subheader("Welcome to the Home Page!")
        st.markdown("""
        <div style="background-color: #f1f1f1; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
            <p style="font-size: 1rem; color: black;">
                This platform is exclusively designed for those with a keen interest in agricultural data, whether you're a researcher, policymaker, farmer, or data enthusiast. Here, you'll find in-depth visualizations and analyses based on <strong>last year's agricultural data</strong> that are crucial for making informed decisions about crop production, pricing, and regional agriculture trends.

                Designed with the agricultural community in mind, this platform helps:
                
                1 - Farmers make better decisions about crop selection based on price and weather data.
                2 - Researchers analyze historical data to explore trends in agriculture.
                3 - Policymakers gain insights into the socio-economic impact of agriculture on different regions.
                4 - Agricultural consultants and industry professionals identify opportunities and challenges in regional markets.
            
            Thank you for using my tool!
            """, unsafe_allow_html=True)


    
    elif option == "Analysis of Crop Prices Across Districts":
        st.subheader("Crop Prices Analysis Across Districts")
        crop_price_analysis()

    elif option == "Yearly Rainfall":
        st.subheader("Seasonwise Rainfall in Tamilnadu Analysis")
        yearlyRainfall()

    elif option == "District wise Rainfall":
        st.subheader("District wise Rainfall")
        districtRainfall()
    elif option == "Peak Sowing and Harvesting Seasons":
        st.subheader("Peak Sowing and Harvesting Seasons")
        peakSowingAndHarvestingSeasons()

    elif option == "Model prediction":
        modelPred()

if __name__ == "__main__":
    main()