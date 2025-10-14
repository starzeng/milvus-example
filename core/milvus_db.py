import os

import dotenv
from pymilvus import MilvusClient

dotenv.load_dotenv()


class MilvusDB:
    def __init__(self):
        self.milvus_client = MilvusClient(
            uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            db_name=os.getenv("MILVUS_DB_NAME", "default"),
            user=os.getenv("MILVUS_USER"),
            password=os.getenv("MILVUS_PASSWORD"),
            token=os.getenv("MILVUS_TOKEN"),
        )

    def get_client(self):
        return self.milvus_client

milvus_db = MilvusDB()