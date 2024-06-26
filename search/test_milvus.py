from pymilvus import connections, Collection

# 建立连接
connections.connect()

collection_name = 'test01'

# 检查集合是否存在
try:
    collection = Collection(collection_name)
    if not collection.is_empty:
        print(f"Collection {collection_name} exists.")
        
        # 检查集合是否有索引
        if collection.has_index():
            print(f"Collection {collection_name} has an index.")
        else:
            print(f"Collection {collection_name} does not have an index.")
    else:
        print(f"Collection {collection_name} does not exist or is empty.")
except Exception as e:
    print(f"Error occurred: {e}")
