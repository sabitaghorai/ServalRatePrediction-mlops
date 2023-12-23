import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("End to End Survival Rate estimation Pipeline with ZenML")

    st.markdown(
        """ 
        #### Problem Statement 
         The objective here is to predict Survival Rate for a given patient details   
        """
    )

 

    with st.sidebar:
        age = st.slider("Age", min_value=18, max_value=100, step=1)
        sex = st.slider("Sex", 0, 1)
        slos = st.slider("Slos")
        d_time = st.slider("D.time")
        dzclass = st.slider("Dzclass", 0, 7)
        dzgroup = st.slider("Dzgroup", 0, 3)
        num_co = st.slider("Num.co", 1, 10)
        scoma = st.slider("Scoma", 1, 100)
        charges = st.slider("Charges", 50000, 100000)
        totcst = st.slider("Totcst", 30000, 3000000)
        avtisst = st.slider("Avtisst", 20, 70)
        race = st.slider("Race",0,3)
        meanbp = st.slider("Meanbp", 0, 10)
        wblc = st.slider("Wblc", 0, 200)
        hrt = st.slider("Hrt", 0, 10)
        resp = st.slider("Resp", 0, 60)
        temp = st.slider("Temp", 30, 40)
        crea = st.slider("Crea", 0, 10)
        sod = st.slider("Sod", 100, 200)
        adlsc = st.slider("Adlsc", 0, 10)

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
            run_main()

        df = pd.DataFrame(
            {
                'age': [age],
                'sex': [sex],
                'slos': [slos],
                'd.time': [d_time],
                'dzgroup': [dzgroup],
                'dzclass': [dzclass],
                'num.co': [num_co],
                'scoma': [scoma],
                'charges': [charges],
                'totcst': [totcst],
                'avtisst': [avtisst],
                'race': [race],
                'meanbp': [meanbp],
                'wblc': [wblc],
                'hrt': [hrt],
                'resp': [resp],
                'temp': [temp],
                'crea': [crea],
                'sod': [sod],
                'adlsc': [adlsc]
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Survival status is  :-{}".format(
                pred
            )
        )

if __name__ == "__main__":
    main()
