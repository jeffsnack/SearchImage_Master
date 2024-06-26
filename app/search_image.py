import numpy as np
from pymilvus import connections, Collection
from search.vgg import VGGNet

def load_collection(collection_name):
    try:
        # 检查集合是否存在
        collection = Collection(collection_name)
        if not collection.is_empty:
            print(f"Collection {collection_name} does exist.")
            collection.load()
            print(f"Collection {collection_name} is loaded.")
            return True
        else:
            print(f"Collection {collection_name} does not exist.")
            return False
    except Exception as e:
        print(f"Error when loading collection {collection_name}: {e}")
        return False

def search(path):
    vgg = VGGNet()

    # 加载集合
    if not load_collection('test01'):
        return []  # 如果集合无法加载，则返回空列表或错误信息

    test_vectors = vgg.extract_feat(path)

    # 设置搜索参数
    search_param = {"metric_type": "L2", "params": {"nprobe": 500}}
    
    # 执行搜索
    try:
        collection = Collection('test01')
        results = collection.search(data=[test_vectors.tolist()], 
                                    anns_field="embedding", 
                                    param=search_param, 
                                    limit=30)
        formatted_results = [{"id": result.id, "distance": result.distance} for result in results[0]]
        return formatted_results
    except Exception as e:
        print(f"Error during search: {e}")
        return []

if __name__ == '__main__':
    # 假设你有一个有效的图片路径
    search("有效的图片路径")
