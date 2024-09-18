import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        # Data Ingestion
        obj=DataIngestion()
        train_data,test_data=obj.initiate_data_ingestion()
        
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

        modeltrainer=ModelTrainer()
        best_model_score = modeltrainer.initiate_model_trainer(train_arr,test_arr)
        
        logging.info(f"Training Pipeline Completed with Best Model Score: {best_model_score}")
        
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_pipeline()
