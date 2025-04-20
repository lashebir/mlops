from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib

class LASSOScoring(FlowSpec):

    data = Parameter('data', default=None, type=str, required=True)
    model_name = Parameter('model_name', default=None, type=str, required=True)
    version = Parameter('version', default=None, type=str, required=False)
    run_id = Parameter('run_id', default=None, type=str, required=True)
    uri = Parameter('uri', default=None, type=str, required=False)

    @step
    def start(self):
        '''
        Load data from the path provided into a dataframe.    
        '''
        self.raw = pd.read_csv(self.data)
        print("Data loaded succesfully")

        self.df_raw = pd.DataFrame(data = self.raw)
        print("Dataframe created")

        self.X = self.df_raw.iloc[:,:-1]

        self.next(self.clean)

    @step
    def clean(self):
        '''
        Cleaning:
        - Dropping N/A values
        - Scaling numeric input (with training scaler)
        - Encoding string input (with training one-hot encoder)
        '''
        self.df_clean = self.df_raw.dropna()
        print("N/A values removed ✅\n")

        scale_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{self.run_id}/preprocessor/scaler_dict.pkl")
        encode_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{self.run_id}/preprocessor/encoder_dict.pkl")

        scalers = joblib.load(scale_path)
        encoders = joblib.load(encode_path)

        for col in self.df_clean.columns.to_list():

            if pd.api.types.is_numeric_dtype(self.df_clean[col]) and col in scalers:
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
                scaler = scalers[col]
                self.df_clean[col] = scaler.transform(self.df_clean[[col]])

            elif self.df_clean[col].dtype == 'object' and col in encoders:
                encoder = encoders[col]
                encoded = encoder.transform(self.df_clean[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=self.df_clean.index)
                self.df_clean.drop(columns = [col], inplace=True)
                self.df_clean = pd.concat([self.df_clean, encoded_df], axis=1)
            
        print("Numeric and string inputs cleaned ✅")

        self.next(self.load_model)

    @step
    def load_model(self):
        '''
        Load model specified by user
        '''
        mlflow.set_tracking_uri(self.uri)
        if self.version:
            self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{self.version}")
        else:
            self.model = mlflow.sklearn.load_model(f"models:/{self.model_name}")
        print("Model loaded successfully!")

        self.next(self.end)

    @step
    def end(self):
        '''
        Run and document prediction on model specified by user
        '''
        now = datetime.now()
        predictions = self.model.predict(self.X)
        pred_df = pd.DataFrame(predictions,columns=['predictions'])
        pred_df.to_csv(f'prediction_{self.model_name}_{now}.csv', index=False)
        print("Model predictions ready for review.")

if __name__ == '__main__':
    LASSOScoring()
