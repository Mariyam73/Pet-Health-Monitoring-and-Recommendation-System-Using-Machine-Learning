import sys

print(sys.executable)
print(sys.version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load models
main_model = joblib.load('model/kmeans.pkl')     # KMeans clustering model
scaler = joblib.load('model/scaler.pkl')         # Scaler for features
rf_model = joblib.load('model/rf_model.pkl')     # Random Forest prediction model

# Define encoders for categorical variables
breed_map = {
    'Australian Shepherd': 0, 'Dachshund': 1, 'Chihuahua': 2, 'Siberian Husky': 3,
    'Boxer': 4, 'Labrador Retriever': 5, 'Bulldog': 6, 'Rottweiler': 7,
    'German Shepherd': 8, 'Golden Retriever': 9, 'Poodle': 10, 'Doberman': 11,
    'Great Dane': 12, 'Beagle': 13, 'Yorkshire Terrier': 14
}
breed_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
sex_map = {'Female': 0, 'Male': 1}
spay_map = {'Spayed': 1, 'Neutered': 1, 'No': 0}
activity_map = {'Low': 0, 'Moderate': 1, 'Active': 2, 'Very Active': 3}
diet_map = {'Home cooked': 0, 'Wet food': 1, 'Hard food': 2, 'Special diet': 3}
bool_map = {'No': 0, 'Yes': 1}
owner_act_map = {'Low': 0, 'Moderate': 1, 'Active': 2, 'Very Active': 3}

# Recommendations per cluster
recommendations_dict = {
    0: [
        "Establish a structured routine to reduce stress and maintain consistency.",
        "Provide a calm, quiet space for restful sleep (at least 11 hours/day).",
        "Use moderate, low-impact exercise daily (e.g., 2 x 15-20 min walks).",
        "Maintain a steady, high-quality diet; consider MCT or omega-3 enriched food.",
        "Introduce omega-3 fatty acids (e.g., fish oil) to support brain and joint health.",
        "Consult vet for seizure logs and schedule semi-annual health checks.",
        "Avoid high-sodium treats if on potassium bromide medication."
    ],
    1: [
        "Ensure daily physical activity (~30-45 min); include walks, fetch, or swimming.",
        "Feed a controlled, nutritious diet with limited treats (use kibble as training rewards).",
        "Use puzzle feeders or slow-feed bowls to provide mental stimulation during meals.",
        "Provide joint support supplements like glucosamine and omega-3s preventively.",
        "Incorporate regular mental enrichment (training, scent games, or toy rotation).",
        "Schedule annual vet exams and routine dental cleanings.",
        "Monitor weight monthly and adjust feeding as needed."
    ],
    2: [
        "Provide 60-120 min of diverse, high-energy exercise (e.g., fetch, running, agility).",
        "Use dog sports or advanced training to channel energy and avoid boredom.",
        "Prevent overheating exercise during cool hours, ensure frequent hydration breaks.",
        "Feed high-protein, high-fat active/performance diets; monitor body condition.",
        "Supplement with omega-3s and consider glucosamine for joint support.",
        "Use paw protection and soft bedding; inspect for injuries after exercise.",
        "Ensure sufficient rest and recovery time with a consistent daily routine."
    ]
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        required_fields = [
            'Breed', 'Breed Size', 'Sex', 'Age', 'Weight (lbs)', 'Spay/Neuter Status',
            'Daily Activity Level', 'Diet', 'Daily Walk Distance (miles)',
            'Other Pets in Household', 'Medications', 'Seizures',
            'Hours of Sleep', 'Play Time (hrs)', 'Owner Activity Level',
            'Annual Vet Visits', 'Average Temperature (F)'
        ]

        missing = [field for field in required_fields if field not in data]
        if missing:
            # return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
            return jsonify({'error': 'Missing fields: ' + ', '.join(missing)}), 400

        # Validate categorical inputs
        if data['Breed'] not in breed_map:
            #return jsonify({'error': f"Invalid breed: {data['Breed']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Breed'])}), 400
        if data['Breed Size'] not in breed_size_map:
            #return jsonify({'error': f"Invalid breed size: {data['Breed Size']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Breed Size'])}), 400
        if data['Sex'] not in sex_map:
            #return jsonify({'error': f"Invalid sex: {data['Sex']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Sex'])}), 400
        if data['Spay/Neuter Status'] not in spay_map:
            #return jsonify({'error': f"Invalid spay/neuter status: {data['Spay/Neuter Status']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Spay/Neuter Status'])}), 400
        if data['Daily Activity Level'] not in activity_map:
            #return jsonify({'error': f"Invalid activity level: {data['Daily Activity Level']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Daily Activity Level'])}), 400
        if data['Diet'] not in diet_map:
            #return jsonify({'error': f"Invalid diet: {data['Diet']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Diet'])}), 400
        if data['Other Pets in Household'] not in bool_map:
            #return jsonify({'error': f"Invalid other pets value: {data['Other Pets in Household']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Other Pets in Household'])}), 400
        if data['Medications'] not in bool_map:
            #return jsonify({'error': f"Invalid medications value: {data['Medications']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Medications'])}), 400
        if data['Seizures'] not in bool_map:
            #return jsonify({'error': f"Invalid seizures value: {data['Seizures']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Seizures'])}), 400
        if data['Owner Activity Level'] not in owner_act_map:
            #return jsonify({'error': f"Invalid owner activity level: {data['Owner Activity Level']}"}), 400
            return jsonify({'error': "Invalid breed: {}".format(data['Owner Activity Level'])}), 400

        # Convert inputs to numeric features
        features = [
            breed_map[data['Breed']],
            breed_size_map[data['Breed Size']],
            sex_map[data['Sex']],
            float(data['Age']),
            float(data['Weight (lbs)']),
            spay_map[data['Spay/Neuter Status']],
            activity_map[data['Daily Activity Level']],
            diet_map[data['Diet']],
            float(data['Daily Walk Distance (miles)']),
            bool_map[data['Other Pets in Household']],
            bool_map[data['Medications']],
            bool_map[data['Seizures']],
            float(data['Hours of Sleep']),
            float(data['Play Time (hrs)']),
            owner_act_map[data['Owner Activity Level']],
            float(data['Annual Vet Visits']),
            float(data['Average Temperature (F)'])
        ]

        # Scale features before clustering
        scaled_features = scaler.transform([features])

        # Predict cluster
        cluster = main_model.predict(scaled_features)[0]

        # Get recommendations for cluster
        recs = recommendations_dict.get(cluster, ["Consult your vet for personalized guidance."])

        # Predict with RF model (assuming it needs unscaled features, adjust if not)
        rf_prediction = rf_model.predict([features])[0]

        return jsonify({
            'cluster': int(cluster),
            'recommendations': recs,
            'rf_model_prediction': int(rf_prediction),
            'status': 'success'
        })

    except ValueError as ve:
        #return jsonify({'error': f'Invalid numeric value: {ve}'}), 400
        return jsonify({'error': 'Invalid numeric value: {}'.format(ve)}), 400
    except Exception as e:
        #return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
        return jsonify({'error': 'Unexpected error: {}'.format(str(e))}), 500


if __name__ == '__main__':
    app.run(debug=True)
