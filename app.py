# Import Library
from google.cloud import aiplatform
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import uuid
import os
import folium
from streamlit_folium import folium_static
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv('API_KEY')

# # Set the path to the JSON file relative to the current working directory
current_directory = os.getcwd()
credentials_path = os.path.join(current_directory, 'credentials.json')

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

project_id = "ctelkom"
location = "us-west1"
model_id = "6405552433283465216"
endpoint_id = "3363546806855139328"

# endpoint
endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id,
                               project=project_id,
                               location=location)

def main():
    st.sidebar.title('Taxi Fare Prediction App Demo')
    st.sidebar.markdown('''
    ## About
    This tool, developed with [Vertex AI AutoML]('https://cloud.google.com/vertex-ai/docs'), 
    enables the estimation of taxi fares in the Chicago, Illinois, USA region. 
    Geolocation is acquired through the [Google Places API]('https://developers.google.com/maps/documentation/places/web-service'),
    while the [Google Distance Matrix API]('https://developers.google.com/maps/documentation/distance-matrix') is utilized to gather information on distance and trip duration.

    ''')

    def app():
        st.header('Chicago Taxi Fare Prediction App 🚕')
        # Generate a random UUID
        taxi_id = str(uuid.uuid4()).replace("-", "")[:32]

        col1, col2 = st.columns(2)
        # Datetime input
        with col1:
            date_input = st.date_input('Select a date: ')
        with col2: 
            time_input = st.time_input('Select time: ')

        combined_datetime = datetime.combine(date_input, time_input)

        st.write('---')

        # Origin and Destination
        def places(api_key, place):
          headers = {
          'Content-Type': 'application/json',
          'X-Goog-Api-Key': api_key,
          'X-Goog-FieldMask': 'places.location',
          }

          data = {
              'textQuery': place,
              'maxResultCount': 1,
          }
          url = 'https://places.googleapis.com/v1/places:searchText'

          # Convert data to JSON format
          json_data = json.dumps(data)

          # Make the POST request
          response = requests.post(url, data=json_data, headers=headers)

          # Print the response
          result = response.json()

          # Convert JSON data to DataFrame
          df_place = pd.json_normalize(result['places'])
          return df_place

                # Trip 
        
        def get_distance_matrix(Origin, Destination, api_key):
            base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': origins,
                'destinations': destinations,
                'key': api_key
            }

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Raise an HTTPError for bad responses

                data = response.json()
                return data

            except requests.exceptions.HTTPError as errh:
                st.write(f"HTTP Error: {errh}")
            except requests.exceptions.ConnectionError as errc:
                st.write(f"Error Connecting: {errc}")
            except requests.exceptions.Timeout as errt:
                st.write(f"Timeout Error: {errt}")
            except requests.exceptions.RequestException as err:
                st.write(f"Request Error: {err}")

            return None
        
        suffix = ",+Chicago,+IL,+USA"

        col1, col2 = st.columns(2)
        with col1:
            Origin = st.text_input("Pickup Location: ",help='[Community areas in Chicago](https://en.wikipedia.org/wiki/Community_areas_in_Chicago)')
            if Origin == '':
                st.warning("Please input your pickup location")
            else:
                df_origin = places(api_key,Origin)
                origins = Origin.replace(' ', '+') + suffix
                pickup_latitude = df_origin['location.latitude'].iloc[0]
                pickup_longitude = df_origin['location.longitude'].iloc[0]

        with col2:
            Destination = st.text_input("Dropoff Location: ",help='[Community areas in Chicago](https://en.wikipedia.org/wiki/Community_areas_in_Chicago)')
            if Destination == '':
                st.warning("Please input your dropoff location")
            else:
                df_destination = places(api_key,Destination)
                destinations = Destination.replace(' ', '+') + suffix
                dropoff_latitude = df_destination['location.latitude'].iloc[0]
                dropoff_longitude = df_destination['location.longitude'].iloc[0]

        # Check if both origins and destinations are defined before making the API request
        if 'origins' in locals() and 'destinations' in locals():
            df = pd.read_csv('taxi-fare.csv')
            min_latitude = df['pickup_latitude'].min()
            max_latitude = df['pickup_latitude'].max()
            min_longitude = df['pickup_longitude'].min()
            max_longitude = df['pickup_longitude'].max()
            Average_Latitude = (min_latitude + max_latitude) / 2
            Average_Longitude = (min_longitude + max_longitude) / 2
            initial_location = [Average_Latitude, Average_Longitude]

            result = get_distance_matrix(origins, destinations, api_key)

            if result and result['status'] == 'OK':
                distance_text = result['rows'][0]['elements'][0]['distance']['text']
                distance_value = result['rows'][0]['elements'][0]['distance']['value']

                duration_text = result['rows'][0]['elements'][0]['duration']['text']
                duration_value = result['rows'][0]['elements'][0]['duration']['value']

                st.write(f"Distance: {distance_text} ({distance_value} meters) & Duration: {duration_text} ({duration_value} seconds)")
            else:
                st.write("Failed to retrieve distance matrix.")

            # Check if any value is outside the coverage area
            if min_latitude <= pickup_latitude <= max_latitude and min_longitude <= pickup_longitude <= max_longitude:
                pickup_latitude = pickup_latitude
                pickup_longitude = pickup_longitude
                st.write("Pickup location is inside our coverage area")
            else:
                st.write("Pickup location is outside our coverage area")

            # trip in seconds
            trip_seconds = duration_value

            # # convert from meters to miles
            trip_miles = (distance_value * 1 / 1609.344)

            # Check if any value is outside the coverage area
            if min_latitude <= pickup_latitude <= max_latitude and min_longitude <= pickup_longitude <= max_longitude:
                pickup_latitude = pickup_latitude
                pickup_longitude = pickup_longitude
                st.write('---')
                st.write("# Maps 🗺️")
                mymap = folium.Map(location = initial_location, zoom_start=10, control_scale=True)

                iframe_origin = folium.IFrame(f'Pickup Location: {Origin}', width=250, height=30)
                popup_origin = folium.Popup(iframe_origin, max_width=250)
                icon_color_origin = 'orange'
                folium.Marker(location=[pickup_latitude,pickup_longitude], popup = popup_origin, icon=folium.Icon(color=icon_color_origin, icon='home')).add_to(mymap)

                iframe_destination = folium.IFrame(f'Dropoff Location: {Destination}', width=250, height=30)
                popup_destination = folium.Popup(iframe_destination, max_width=250)
                icon_color_destination = 'green'
                folium.Marker(location=[dropoff_latitude,dropoff_longitude], popup = popup_destination, icon=folium.Icon(color=icon_color_destination, icon='flag')).add_to(mymap)
                folium_static(mymap)
                st.write('---')
                st.write(" ## Order Summary:")
                st.write(f" Taxi ID: {taxi_id}")
                st.write(f" Trip start time: {combined_datetime}")
                st.write(f" Pickup Location: {Origin}")
                st.write(f" Dropoff Location: {Destination}")
                st.write(f" Trip distance: {distance_text}")
                st.write(f" Trip duration: {duration_text}")
            else:
                st.write("Pickup location is outside our coverage area")

            # Predict using Endpoint
            data = {
                'pickup_latitude': [pickup_latitude],
                'pickup_longitude': [pickup_longitude],
                'dropoff_latitude': [dropoff_latitude],
                'dropoff_longitude': [dropoff_longitude],
                'trip_seconds': [trip_seconds],
                'trip_miles': [trip_miles,]
            }

            df_prediction = pd.DataFrame(data)
            df_prediction['trip_seconds'] = df_prediction['trip_seconds'].astype(str)

            # Prepare instances
            instances = df_prediction.to_dict(orient='records')

            # Send prediction request
            response = endpoint.predict(instances=instances)

            # Extract predictions
            predictions = response.predictions

            # Create a new DataFrame with predictions
            predictions_df = pd.DataFrame(predictions)

            st.success(f"## Total cost of the trip: $ {round(predictions_df['value'].iloc[0],2)}")

    def eda():
        st.header('Exploratory Data Analysis')
        df = pd.read_csv('taxi-fare.csv')
        df.drop('Unnamed: 0',axis=1,inplace=True)
        df["trip_hours"] = round(df["trip_seconds"] / 3600, 2)
        df["trip_speed"] = round(df["trip_miles"] / df["trip_hours"], 2)
        selection = st.selectbox('Select Here: ',['Dataset',"Histograms and Boxplots","Trip Duration & Trip Speed","Relationship Between Variable"])

        if selection == 'Dataset':
            st.write(" ## Dataset")
            st.write(' Dataset obtained from Google Cloud Platform - BigQuery database : `chicago_taxi_trips`, table: `taxi_trips`.')
            st.dataframe(df)
            st.write(''' The chosen dataset consists of the following fields:

    - `taxi_id` : A unique identifier for the taxi.
    - `trip_start_timestamp`: When the trip started, rounded to the nearest 15 minutes.
    - `trip_seconds`: Time of the trip in seconds.
    - `trip_miles`: Distance of the trip in miles.
    - `pickup_latitude`: The latitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy.
    - `pickup_longitude`: The longitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy.
    - `dropoff_latitude`: The latitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy.
    - `dropoff_longitude`: The longitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy.
    - `trip_total`: Total cost of the trip, the total of the fare, tips, tolls, and extras.        

            ''')
            st.write('')
            st.write('Numerical Distributions of the fields')
            df.describe().T

        elif selection == "Histograms and Boxplots":
            st.write("# Histograms and Boxplots")
            target = "trip_total"
            num_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', "trip_seconds", "trip_miles"]
            for i in num_cols + [target]:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                df[i].plot(kind="hist", bins=100, ax=ax[0])
                ax[0].set_title(str(i) + " -Histogram")
                df[i].plot(kind="box", ax=ax[1])
                ax[1].set_title(str(i) + " -Boxplot")
                st.pyplot(fig)

                # Calculate IQR
                Q1 = df[i].quantile(0.25)
                Q3 = df[i].quantile(0.75)
                IQR = Q3 - Q1

                # Count outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)][i]

                # Print the number of outliers and percentage
                num_outliers = len(outliers)
                total_values = len(df[i])
                percentage_outliers = (num_outliers / total_values) * 100

                st.write("Q1: ",Q1)
                st.write("Q3: ",Q3)
                st.write(f"Number of outliers in {i}: {len(outliers)}")
                st.write(f"Percentage of outliers in {i}: {percentage_outliers:.2f}%")
                st.write("---")

        elif selection == "Trip Duration & Trip Speed":
            st.write("# Trip Duration and Trip Speed")
            st.write("The field `trip_seconds` describes the time taken for the trip in seconds. For ease of our analysis, let us convert it into hours.")
                       
            fig, ax = plt.subplots()
            df["trip_hours"].plot(kind="box", ax=ax)
            ax.set_title("Trip Hours Boxplot")
            st.pyplot(fig)

            st.write('---')
            st.write("trip_speed can be added by dividing `trip_miles` and `trip_hours` to understand the speed of the trip in miles/hour")
                        
            fig, ax = plt.subplots()
            df["trip_speed"].plot(kind="box", ax=ax)
            ax.set_title("Trip Speed Boxplot")
            st.pyplot(fig)
            st.write("From the box plots and the histograms visualized so far, it is evident that there are some outliers causing skewness in the data. The presence of outliers can have an impact on the model's performance and accuracy.")
        
        else:
            st.title('Relationship between variable')
            st.write('To better understand the relationship between the variables, a pair-plot can be plotted.')
            try:
            # Create a pairplot for a sample of 10,000 rows
                pairplot = sns.pairplot(data=df[["trip_seconds", "trip_miles", "trip_total", "trip_speed"]].sample(10000))
            
            # Display the pairplot in Streamlit
                st.pyplot(pairplot)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
            st.write('You can see some linear relationships between the independent variables considered in the pair-plot. For example, `trip_miles` and the dependant variable `trip_total`')
            
    def model():
        st.title("Model Evaluation")
        st.write("## Metric")
        st.image('model_evaluation.png',caption='To zoom in click view full screen buton')
        st.write(''' 
- `MAE` = 2.709, Mean absolute error (MAE) is the average of absolute differences between observed and predicted values. A low value indicates a higher-quality model, where 0 means the model made no errors. Interpreting MAE depends on the range of values in the series. MAE has the same unit as the target column.
- `MAPE` = 14.598%, The mean absolute percentage error (MAPE) is the average of absolute percentage errors. MAPE ranges from 0% to 100%, where a lower value indicates a higher quality model. MAPE becomes infinity if 0 values are present in the ground truth data.
- `RMSE` = 5.542, Root mean square error (RMSE) is the root of squared differences between observed and predicted values. A lower value indicates a higher quality model, where 0 means the model made no errors. Interpreting RMSE depends on the range of values in the series. RMSE is more responsive to large errors than MAE.                  
- `RMSLE` = 0.198, Root mean squared log error (RMSLE) is the root of squared averages of log differences between observed and predicted values. Interpreting RMSLE depends on the range of values in the series. RMSLE is less responsive to outliers than RMSE, and it tends to penalise underestimations slightly more than overestimations.
- `r^2` = 90.6%, R squared (R^2) is the square of the Pearson correlation coefficient between the observed and predicted values. This ranges from 0 to 1, where a higher value indicates a higher-quality model.
                 
Based on the evaluation metrics above, it can be said that this model is good enough to predict taxi fares.
''')
        st.write('---')
        st.write("## Feature Importance")
        st.image('feature_importance.png',caption='To zoom in click view full screen buton')
        st.write('Model feature attribution tells you how important each feature is when making a prediction. Attribution values are expressed as a percentage; the higher the percentage, the more strongly that feature impacts a prediction on average. Model feature attribution is expressed using the sampled Shapley method.')
    selected_option  = st.sidebar.radio("Option: ", ["Application 🚕", "EDA 📊", "Model Evaluation"])
    if selected_option == "Application 🚕":
        app()
    elif selected_option == 'EDA 📊':
        eda()
    else:
        model()

    st.sidebar.markdown(''' 
                    ## Created by: 
                    Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
                    ''')
    
if __name__ == '__main__':
    main()