import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("Employee Attrition Prediction")

    # Define your list of columns
    columns_for_df = ['sex_female', 'sex_male', 'dzgroup_ARF/MOSF w/Sepsis', 'dzgroup_CHF',
       'dzgroup_COPD', 'dzgroup_Cirrhosis', 'dzgroup_Colon Cancer',
       'dzgroup_Coma', 'dzgroup_Lung Cancer', 'dzgroup_MOSF w/Malig',
       'dzclass_ARF/MOSF', 'dzclass_COPD/CHF/Cirrhosis', 'dzclass_Cancer',
       'dzclass_Coma', 'race_asian', 'race_black', 'race_hispanic',
       'race_other', 'race_white', 'race_nan', 'age', 'slos',
       'd.time', 'num.co', 'scoma', 'charges', 'totcst', 'avtisst', 'meanbp',
       'wblc', 'hrt', 'resp', 'temp', 'crea', 'sod', 'adlsc']
    # Create a dictionary to store the selected inputs
    selected_data = {col: 0 for col in columns_for_df}

    # Create a dictionary mapping each categorical feature to its options
    categorical_options = {
    'gender': ['male', 'female'],
    'dzgroup': ['Lung Cancer', 'Colon Cancer', 'ARF/MOSF w/Sepsis', 'MOSF w/Malig', 'Cirrhosis', 'CHF', 'COPD', 'Coma'],
    'dzclass': ['Cancer', 'ARF/MOSF', 'COPD/CHF/Cirrhosis', 'Coma'],
    'race': ['black', 'hispanic', 'white', 'other', 'asian']

}

    # Create select boxes for categorical features
    for feature, options in categorical_options.items():
        selected_value = st.sidebar.selectbox(f"Select {feature}:", options)
        selected_data[feature + "_" + selected_value] = 1

    # Define numerical features and their corresponding slider ranges
    numerical_features = {
    'age': (18, 100, 30),
    'slos': (3, 241, 0),
    'd.time': (3, 2029, 0),
    'num.co': (0, 7, 0),
    'scoma': (0, 100, 0),
    'charges': (1635, 740010, 0),
    'totcst': (0, 390460, 0),
    'avtisst': (1, 64, 0),
    'meanbp': (0, 180, 0),
    'wblc': (0, 100, 0),
    'hrt': (0, 300, 0),
    'resp': (0, 64, 0),
    'temp': (32, 40, 0),
    'crea': (0, 10, 0),
    'sod': (118, 175, 0),
    'adlsc': (0, 7, 0)
    }

    
    


    # Create sliders for numerical features
    for feature, (min_val, max_val, default_val) in numerical_features.items():
        selected_data[feature] = st.sidebar.slider(feature, min_val, max_val, default_val)

    # Create a DataFrame with the selected inputs
    data_df = pd.DataFrame([selected_data])

    # Display the selected inputs
    st.write("Selected Inputs:")
    st.write(data_df)
    print(data_df)

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            main()
        json_list = json.loads(json.dumps(list(data_df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Predicted Employee Attrition Probability (0 - 1): {:.2f}".format(
                pred[0]
            )
        )

if __name__ == "__main__":
    main()
