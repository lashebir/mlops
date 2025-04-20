from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

class LASSOTrainer(FlowSpec):

    data = Parameter('data', default=None, type=str, required=True)
    k = Parameter('k', default=5, type=int, required=True)
    random_state = Parameter('random_state', default=42, type=int, required=True)

    @step
    def start(self):
        '''
        Load data from the path provided into a dataframe.    
        '''
        self.raw = pd.read_csv(self.data)
        print("Data loaded succesfully")

        self.df_raw = pd.DataFrame(data = self.raw)
        print("Dataframe created")

        self.next(self.clean)
    @step
    def clean(self):
        '''
        Cleaning:
        - Dropping N/A values
        - Scaling numeric input
        - Encoding string input (one-hot encoder)
        '''
        self.df_clean = self.df_raw.dropna()
        print("N/A values removed ✅\n")
        
        scalers = {}
        encoders = {}

        for col in self.df_clean.iloc[:,:-1].columns.to_list():

            if pd.api.types.is_numeric_dtype(self.df_clean[col]):
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
                scaler = StandardScaler()
                self.df_clean[col] = scaler.fit_transform(self.df_clean[[col]])
                scalers[col] = scaler

            elif self.df_clean[col].dtype == 'object':
                encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(self.df_clean[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=self.df_clean.index)
                self.df_clean.drop(columns = [col], inplace=True)
                self.df_clean = pd.concat([self.df_clean, encoded_df], axis=1)
                encoders[col] = encoder
        
        self.scaler_path = "scaler_dict.pkl"
        joblib.dump(scalers, self.scaler_path)

        self.encoder_path = "encoder_dict.pkl"
        joblib.dump(encoders, self.encoder_path)

        print("Numeric and string inputs cleaned ✅")

        self.next(self.tuning)

    @step
    def tuning(self):
        '''
        Running grid search to find best lambda for LASSO model
        '''
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        self.X = self.df_clean.iloc[:, :-1]
        self.y = self.df_clean.iloc[:,-1]
        self.best_lambda = 0
        self.best_score = 0
        for train, val in kf.split(self.X):
            for alpha in np.arange(0.001, 0.1, 0.01):
                model = Lasso(alpha=alpha)
                model.fit(self.X.iloc[train], self.y.iloc[train])
                score = model.score(self.X.iloc[val], self.y.iloc[val])
                if self.best_score < score:
                    self.best_score = score
                    self.best_lambda = alpha
        
        print(f"Best lambda found: {self.best_lambda}")

        self.next(self.end)

    @step
    def end(self):
        '''
        Use best lambda from the tuning stage to train and register the final model, exported to MLFlow
        '''
        self.model = Lasso(alpha=self.best_lambda)
        self.model.fit(self.X, self.y)
        print("Model trained successfully")

        with mlflow.start_run() as run:
            uri = mlflow.get_tracking_uri()
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("lambda", self.best_lambda)
            mlflow.log_artifact(self.scaler_path, artifact_path="preprocessor")
            mlflow.log_artifact(self.encoder_path, artifact_path="preprocessor")
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(
                model_uri=model_uri,  # Path to model artifact
                name="trained_model"
    )
            print(f"Model registered: {result.name}, version: {result.version}, run_id: {run.info.run_id}, uri: {uri}")

if __name__ == '__main__':
    LASSOTrainer()
    
        



