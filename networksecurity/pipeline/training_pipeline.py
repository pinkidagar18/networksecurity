import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

# Import constants if S3 sync is needed
# Uncomment these lines if you want to use S3
# from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
# from networksecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
        # Initialize S3 sync only if using cloud storage
        # Uncomment this if you want to use S3
        # self.s3_sync = S3Sync()
        
        # Flag to control S3 sync (set to False for local-only training)
        self.use_s3_sync = False  # Change to True if you want S3 sync
        

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            logging.info("Starting data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            logging.info("Starting data transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            logging.info("Starting model training")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    ## Local artifact is going to S3 bucket (Optional)
    def sync_artifact_dir_to_s3(self):
        """
        Sync local artifacts to S3 bucket
        Only runs if use_s3_sync is True
        """
        try:
            if not self.use_s3_sync:
                logging.info("S3 sync disabled. Skipping artifact sync to S3.")
                return
            
            # Uncomment these lines if you want to use S3
            # from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
            # aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            # self.s3_sync.sync_folder_to_s3(
            #     folder=self.training_pipeline_config.artifact_dir,
            #     aws_bucket_url=aws_bucket_url
            # )
            # logging.info(f"Artifacts synced to S3: {aws_bucket_url}")
            
            logging.info("S3 artifact sync skipped (not configured)")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    ## Local final model is going to S3 bucket (Optional)
    def sync_saved_model_dir_to_s3(self):
        """
        Sync local final model to S3 bucket
        Only runs if use_s3_sync is True
        """
        try:
            if not self.use_s3_sync:
                logging.info("S3 sync disabled. Skipping model sync to S3.")
                return
            
            # Uncomment these lines if you want to use S3
            # from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
            # aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            # self.s3_sync.sync_folder_to_s3(
            #     folder=self.training_pipeline_config.model_dir,
            #     aws_bucket_url=aws_bucket_url
            # )
            # logging.info(f"Model synced to S3: {aws_bucket_url}")
            
            logging.info("S3 model sync skipped (not configured)")
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def run_pipeline(self):
        """
        Run the complete training pipeline
        Steps:
        1. Data Ingestion - Load data from MongoDB
        2. Data Validation - Validate schema and detect drift
        3. Data Transformation - Apply preprocessing
        4. Model Training - Train and evaluate models
        5. S3 Sync (optional) - Upload artifacts to cloud
        """
        try:
            logging.info("=" * 80)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 80)
            
            # Step 1: Data Ingestion
            logging.info("Step 1/4: Data Ingestion")
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            logging.info("Step 2/4: Data Validation")
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
            # Step 3: Data Transformation
            logging.info("Step 3/4: Data Transformation")
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            
            # Step 4: Model Training
            logging.info("Step 4/4: Model Training")
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Optional: Sync to S3 (only if enabled)
            if self.use_s3_sync:
                logging.info("Syncing artifacts to S3...")
                self.sync_artifact_dir_to_s3()
                self.sync_saved_model_dir_to_s3()
            else:
                logging.info("S3 sync disabled. Models saved locally only.")
            
            logging.info("=" * 80)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            
            return model_trainer_artifact
            
        except Exception as e:
            logging.error("=" * 80)
            logging.error("TRAINING PIPELINE FAILED")
            logging.error("=" * 80)
            raise NetworkSecurityException(e, sys)