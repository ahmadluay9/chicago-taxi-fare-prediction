# Chicago Taxi Fare Prediction Application using Vertex AI AutoML

- [Application Demo](https://taxi-fare-prediction-application-7emkch5d3q-uc.a.run.app/)

The "Chicago Taxi Fare Prediction" project aims to develop a machine learning model using [Vertex AI AutoML](https://cloud.google.com/vertex-ai/docs) that can be used for taxi fare prediction in the Chicago area. The location was obtained using the [Google Places API](https://developers.google.com/maps/documentation/places/web-service), while the distance and duration were obtained using the [Google Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix).

---

## File Explanation
This repository consists of several files :

```
    ┌── .gitignore
    ├── README.md
    ├── app.py
    ├── dockerfile
    ├── requirements.txt
    └── taxi-fare.csv
```
- `README.md`: This is a Markdown file that typically contains documentation for the project. It include information on how to set up and run your application, dependencies, and any other relevant details.
  
- `app.py`: This file is the main script for the frontend of the application and is developed using the Streamlit framework.

- `dockerfile`: Dockerfile is used to build a Docker image for frontend application. It includes instructions on how to set up the environment and dependencies needed for frontend.

- `requirements.txt`: This file lists the Python dependencies required for frontend application. These dependencies can be installed using a package manager like pip.

- `README.md`: This is a Markdown file that typically contains documentation for the project. It include information on how to set up and run your application, dependencies, and any other relevant details.

- `taxi-fare.csv`: This is the CSV file used as the dataset in this project. Dataset obtained from Google Cloud Platform - BigQuery database : `chicago_taxi_trips`, table: `taxi_trips`.
  
---

## Application

### How to use


Users can use this application by entering the desired location on the widget. You can also see the Exploratory Data Analysis and Model Evaluation.

**Preview**

https://github.com/ahmadluay9/chicago-taxi-fare-prediction/assets/123846438/ac9aa27e-93c0-417e-b1d4-b5b02bbeaf68


