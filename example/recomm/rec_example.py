from pymilvus import DataType

from core.milvus_db import milvus_db

milvus_client = milvus_db.get_client()

COLLECTION_NAME="rec"

schema = milvus_client.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)

index_params = milvus_client.prepare_index_params()

index_params.add_index(
    field_name="my_id",
    index_type="AUTOINDEX"
)

index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

res = milvus_client.get_load_state(
    collection_name=COLLECTION_NAME
)

print(res)
