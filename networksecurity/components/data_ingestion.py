import os
import sys
import numpy as np
import pandas as pd
import pymongo

from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB and return as DataFrame
        """
        try:
            logging.info("Connecting to MongoDB")

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            database = mongo_client[self.data_ingestion_config.database_name]
            collection = database[self.data_ingestion_config.collection_name]

            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                raise ValueError("MongoDB collection returned empty dataset")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            logging.info("Successfully fetched data from MongoDB")
            return df

        except Exception as e:
            logging.error("Error while exporting MongoDB collection as dataframe")
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Data exported to feature store successfully")
            return dataframe

        except Exception as e:
            logging.error("Error while exporting data into feature store")
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            logging.info("Performing train-test split")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train-test split completed and files saved")

        except Exception as e:
            logging.error("Error while splitting data into train and test")
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process")

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            logging.error("Data ingestion failed")
            raise NetworkSecurityException(e, sys)
