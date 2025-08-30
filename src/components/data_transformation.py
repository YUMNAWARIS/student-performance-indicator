import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation and data scaling.
        
        '''

        #  0   Hours_Studied               6607 non-null   int64 
        #  1   Attendance                  6607 non-null   int64 
        #  2   Parental_Involvement        6607 non-null   object
        #  3   Access_to_Resources         6607 non-null   object
        #  4   Extracurricular_Activities  6607 non-null   object
        #  5   Sleep_Hours                 6607 non-null   int64 
        #  6   Previous_Scores             6607 non-null   int64 
        #  7   Motivation_Level            6607 non-null   object
        #  8   Internet_Access             6607 non-null   object
        #  9   Tutoring_Sessions           6607 non-null   int64 
        #  10  Family_Income               6607 non-null   object
        #  11  Teacher_Quality             6607 non-null   object
        #  12  School_Type                 6607 non-null   object
        #  13  Peer_Influence              6607 non-null   object
        #  14  Physical_Activity           6607 non-null   int64 
        #  15  Learning_Disabilities       6607 non-null   object
        #  16  Parental_Education_Level    6607 non-null   object
        #  17  Distance_from_Home          6607 non-null   object
        #  18  Gender                      6607 non-null   object
        #  19  Exam_Score                  6607 non-null   int64 


        try:
            numerical_columns = [   
                "Hours_Studied", 
                "Attendance", 
                "Sleep_Hours", 
                "Previous_Scores", 
                "Tutoring_Sessions",
                "Physical_Activity", 
                "Exam_Score"
            ]
            categorical_columns = [
                "Gender",
                "Parental_Involvement",
                "Access_to_Resources",
                "Extracurricular_Activities",
                "Motivation_Level",
                "Internet_Access",
                "Family_Income",
                "Teacher_Quality",
                "School_Type",
                "Peer_Influence",
                "Learning_Disabilities",
                "Parental_Education_Level",
                "Distance_from_Home",
                "Gender",
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Exam_Score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

