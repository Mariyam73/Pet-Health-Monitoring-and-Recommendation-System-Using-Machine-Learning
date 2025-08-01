
from shiny import App, render, ui, reactive
import requests
import pandas as pd
import joblib

# Load local KMeans and scaler for recommendation
kmeans = joblib.load("../Models/kmeans.pkl")
scaler = joblib.load("../Models/scaler.pkl")

# Import the recommender
from recommender import recommend_care_updated

# UI layout
app_ui = ui.page_fluid(
    ui.h2("🐶 Pet Health Prediction & Recommendation"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("Breed", "Breed", {
                "Beagle": "Beagle",
                "Labrador Retriever": "Labrador Retriever",
                "Poodle": "Poodle"
            }),
            ui.input_select("Breed_Size", "Breed Size", ["Small", "Medium", "Large"]),
            ui.input_select("Sex", "Sex", ["Female", "Male"]),
            ui.input_slider("Age", "Age (years)", 0, 20, 7),
            ui.input_numeric("Weight", "Weight (lbs)", 30),
            ui.input_select("Spay", "Spay/Neuter Status", ["No", "Spayed", "Neutered"]),
            ui.input_select("Activity", "Daily Activity Level", ["Low", "Moderate", "Active", "Very Active"]),
            ui.input_select("Diet", "Diet Type", ["Home cooked", "Wet food", "Hard food", "Special diet"]),
            ui.input_slider("Walk", "Daily Walk Distance (miles)", 0, 10, 2),
            ui.input_select("Other_Pets", "Other Pets in Household", ["No", "Yes"]),
            ui.input_select("Meds", "On Medications?", ["No", "Yes"]),
            ui.input_select("Seizures", "Seizures?", ["No", "Yes"]),
            ui.input_slider("Sleep", "Hours of Sleep", 4, 15, 11),
            ui.input_slider("Play", "Play Time (hrs)", 0, 8, 1),
            ui.input_select("Owner_Act", "Owner Activity Level", ["Low", "Moderate", "Active", "Very Active"]),
            ui.input_slider("Vet_Visits", "Annual Vet Visits", 0, 4, 1),
            ui.input_numeric("Temp", "Average Temperature (F)", 70),
            ui.input_action_button("submit", "Submit")
        ),
        ui.panel_main(
            ui.output_text("prediction"),
            ui.output_ui("recommendations")
        )
    )
)

# Server logic
def server(input, output, session):
    @reactive.event(input.submit)
    def _():
        # Encoders
        enc = {
            "Breed Size": {"Small": 0, "Medium": 1, "Large": 2},
            "Sex": {"Female": 0, "Male": 1},
            "Spay": {"No": 0, "Spayed": 1, "Neutered": 1},
            "Activity": {"Low": 0, "Moderate": 1, "Active": 2, "Very Active": 3},
            "Diet": {"Home cooked": 0, "Wet food": 1, "Hard food": 2, "Special diet": 3},
            "Other_Pets": {"No": 0, "Yes": 1},
            "Meds": {"No": 0, "Yes": 1},
            "Seizures": {"No": 0, "Yes": 1},
            "Owner_Act": {"Low": 0, "Moderate": 1, "Active": 2, "Very Active": 3}
        }

        # Prepare raw + encoded input
        pet_input = {
            "Breed": input.Breed(),  # raw name, optional mapping later
            "Breed Size": enc["Breed Size"][input.Breed_Size()],
            "Sex": enc["Sex"][input.Sex()],
            "Age": input.Age(),
            "Weight (lbs)": input.Weight(),
            "Spay/Neuter Status": enc["Spay"][input.Spay()],
            "Daily Activity Level": enc["Activity"][input.Activity()],
            "Diet": enc["Diet"][input.Diet()],
            "Daily Walk Distance (miles)": input.Walk(),
            "Other Pets in Household": enc["Other_Pets"][input.Other_Pets()],
            "Medications": enc["Meds"][input.Meds()],
            "Seizures": enc["Seizures"][input.Seizures()],
            "Hours of Sleep": input.Sleep(),
            "Play Time (hrs)": input.Play(),
            "Owner Activity Level": enc["Owner_Act"][input.Owner_Act()],
            "Annual Vet Visits": input.Vet_Visits(),
            "Average Temperature (F)": input.Temp()
        }

        # Call Flask backend
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=pet_input
        )
        prediction = response.json()["prediction"]

        output.prediction.set(f"📋 Predicted Health Status: {'Healthy ✅' if prediction == 1 else 'Not Healthy ❌'}")

        # Recommendation
        _, recs = recommend_care_updated(pet_input, scaler, kmeans)
        output.recommendations.set(ui.tags.ul([ui.tags.li(r) for r in recs]))

# Create the app
app = App(app_ui, server)

