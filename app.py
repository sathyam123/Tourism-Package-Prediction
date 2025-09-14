import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import gradio as gr
import os

# Set the HF_HUB_CACHE environment variable to a writable directory
os.environ['HF_HUB_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

# Define the model repository ID and filename
model_repo_id = "sathyam123/gradient-boosting-tourism-package-predictor"
filename = "model.joblib"

# Download and load the model
try:
    model_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
    loaded_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None # Handle cases where model loading fails

# Define the exact column names from the training data
# This is crucial for ensuring the input data has the same structure
# as the training data after one-hot encoding.
# You need to manually get this list from your training data preprocessing step.
# Based on the previous notebook state, X_train.columns had 36 columns.
# We will list them out here.
train_columns = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome', 'TypeofContact_Company Invited',
    'TypeofContact_Self Enquiry', 'Occupation_Free Lancer', 'Occupation_Government Sector',
    'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Fe Male',
    'Gender_Female', 'Gender_Male', 'ProductPitched_Basic', 'ProductPitched_Deluxe',
    'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'Designation_AVP', 'Designation_Executive', 'Designation_Manager',
    'Designation_Senior Manager', 'Designation_VP', '__index_level_0__', 'PreferredFoodStar_2.0'
]


def predict_tourism_package(Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched, PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting, Designation, MonthlyIncome):
    """
    Predicts whether a tourism package will be taken based on input features.

    Args:
        (Individual input features as defined in the Gradio interface)

    Returns:
        str: Prediction result ('Taken' or 'Not Taken') or an error message.
    """
    if loaded_model is None:
        return "Error: Model not loaded."

    # Create a dictionary from the input arguments
    input_data = {
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Identify categorical columns (assuming object dtype indicates categorical)
    categorical_cols = input_df.select_dtypes(include=['object']).columns

    # Apply one-hot encoding
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # Align columns with training data - add missing columns with 0 and reorder
    missing_in_input = set(train_columns) - set(input_df.columns)
    for c in missing_in_input:
        input_df[c] = 0

    # Ensure the order of columns is the same as in the training data
    input_df = input_df[train_columns]

    # Make prediction
    prediction = loaded_model.predict(input_df)[0]

    # Return prediction result
    return 'Taken' if prediction == 1 else 'Not Taken'

# Define Gradio Interface
# Determine choices for categorical features based on your data exploration
contact_choices = ['Self Enquiry', 'Company Invited']
occupation_choices = ['Salaried', 'Company Invited', 'Free Lancer', 'Government Sector', 'Small Business']
gender_choices = ['Female', 'Male', 'Fe Male']
product_choices = ['Deluxe', 'Basic', 'Superior', 'Standard', 'King', 'Super Deluxe'] # Corrected based on notebook data
marital_choices = ['Single', 'Divorced', 'Married']
designation_choices = ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'] # Corrected based on notebook data


if loaded_model is not None:
    interface = gr.Interface(
        fn=predict_tourism_package,
        inputs=[
            gr.Number(label="Age"),
            gr.Dropdown(label="Type of Contact", choices=contact_choices),
            gr.Number(label="City Tier"),
            gr.Number(label="Duration of Pitch"),
            gr.Dropdown(label="Occupation", choices=occupation_choices),
            gr.Dropdown(label="Gender", choices=gender_choices),
            gr.Number(label="Number of Persons Visiting"),
            gr.Number(label="NumberOfFollowups"),
            gr.Dropdown(label="Product Pitched", choices=product_choices),
            gr.Number(label="Preferred Property Star"),
            gr.Dropdown(label="Marital Status", choices=marital_choices),
            gr.Number(label="Number of Trips"),
            gr.Radio(label="Passport", choices=[0, 1]),
            gr.Number(label="Pitch Satisfaction Score"),
            gr.Radio(label="Own Car", choices=[0, 1]),
            gr.Number(label="NumberOfChildrenVisiting"),
            gr.Dropdown(label="Designation", choices=designation_choices),
            gr.Number(label="Monthly Income")
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Tourism Package Prediction",
        description="Predict whether a customer will take a tourism package."
    )

    # Launch the app
    interface.launch(server_name="0.0.0.0")
else:
    print("Gradio interface not launched because the model failed to load.")
