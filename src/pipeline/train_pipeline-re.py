import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline.")

            # ---------------------------
            # Data Ingestion
            # ---------------------------
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed.")

            # ---------------------------
            # Data Transformation
            # ---------------------------
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            logging.info("Data transformation completed.")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # ---------------------------
            # Model Training
            # ---------------------------
            model_trainer = ModelTrainer()
            r2_score_value = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info("Model training completed.")
            logging.info(f"Final R2 score: {r2_score_value}")

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print("Training pipeline completed.")
    print("R2 score:", score)