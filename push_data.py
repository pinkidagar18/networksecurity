import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = self.mongo_client[database]
            coll = db[collection]
            coll.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    try:
        # Input details
        FILE_PATH = "Network_data/phisingData.csv"
        DATABASE = "PINKI"
        COLLECTION = "Networkdata"

        # Create an object and perform operations
        network_obj = NetworkDataExtract()
        records = network_obj.csv_to_json_convertor(file_path=FILE_PATH)
        print(records)
        no_of_records = network_obj.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(no_of_records)
    except Exception as e:
        print(f"An error occurred: {e}")
