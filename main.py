import sys

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


def main():
    try:
        logging.info("🚀 Network Security Training Pipeline Started")

        # =========================
        # Training Pipeline Config
        # =========================
        training_pipeline_config = TrainingPipelineConfig()

        # =========================
        # Data Ingestion
        # =========================
        logging.info("📥 Data Ingestion Started")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("✅ Data Ingestion Completed")
        print(data_ingestion_artifact)

        # =========================
        # Data Validation
        # =========================
        logging.info("🔍 Data Validation Started")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("✅ Data Validation Completed")
        print(data_validation_artifact)

        # =========================
        # Data Transformation
        # =========================
        logging.info("🔄 Data Transformation Started")
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("✅ Data Transformation Completed")
        print(data_transformation_artifact)

        # =========================
        # Model Training
        # =========================
        logging.info("🤖 Model Training Started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("✅ Model Training Completed")
        print(model_trainer_artifact)

        logging.info("🎉 Network Security Training Pipeline Completed Successfully")

    except Exception as e:
        logging.error("❌ Error occurred in Network Security Training Pipeline")
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    main()
