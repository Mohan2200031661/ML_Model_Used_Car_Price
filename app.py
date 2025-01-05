from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and any necessary preprocessing files
rf_model = joblib.load('XGBoost_model.pkl')

csv_file_path = 'used_cars_data.csv'  # Path to your dataset
data = pd.read_csv(csv_file_path).replace('?', None)

# Extract unique values for dropdowns
makes = data['make'].dropna().unique().tolist()
fuel_types = data['fuel-type'].dropna().unique().tolist()
num_of_doors = data['num-of-doors'].dropna().unique().tolist()
body_styles = data['body-style'].dropna().unique().tolist()
drive_wheels = data['drive-wheels'].dropna().unique().tolist()
engine_types = data['engine-type'].dropna().unique().tolist()
num_of_cylinders = data['num-of-cylinders'].dropna().unique().tolist()


@app.route('/')
def index():
    """Render the main form with dropdown options."""
    return render_template(
        'index.html',
        makes=makes,
        fuel_types=fuel_types,
        num_of_doors=num_of_doors,
        body_styles=body_styles,
        drive_wheels=drive_wheels,
        engine_types=engine_types,
        num_of_cylinders=num_of_cylinders
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return predicted car price."""
    try:
        # Collect form data
        form_data = request.form

        # Convert form data into a DataFrame for prediction
        input_data = pd.DataFrame({
            "make": [form_data.get("make")],
            "fuel-type": [form_data.get("fuel-type")],
            "num-of-doors": [form_data.get("num-of-doors")],
            "body-style": [form_data.get("body-style")],
            "drive-wheels": [form_data.get("drive-wheels")],
            "curb-weight": [float(form_data.get("curb-weight"))],
            "engine-type": [form_data.get("engine-type")],
            "num-of-cylinders": [form_data.get("num-of-cylinders")],
            "engine-size": [float(form_data.get("engine-size"))],
            "horsepower": [float(form_data.get("horsepower"))],
            "peak-rpm": [float(form_data.get("peak-rpm"))],
            "city-L/100km": [float(form_data.get("city-L/100km"))],
            "highway-mpg": [float(form_data.get("highway-mpg"))]
        })

        # One-hot encoding for categorical features (same as training preprocessing)
        input_data = pd.get_dummies(input_data)

        # Align input_data with model features (fill missing columns with 0)
        model_features = rf_model.feature_names_in_  # Assuming your model exposes this attribute
        input_data = input_data.reindex(columns=model_features, fill_value=0)

        # Make prediction
        prediction = rf_model.predict(input_data)[0]

        # Ensure prediction is a standard float type before returning it
        prediction = float(prediction)

        # Return prediction as JSON
        print(prediction)
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
