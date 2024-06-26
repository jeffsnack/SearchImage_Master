import os
from flask import Flask
from app import app
from time import time, sleep
import numpy as np
from app import db, Image, app_milvus
from search.vgg import VGGNet
from pymilvus import Milvus, DataType, Index, connections,Collection
from PIL import Image as Im

def create_milvus():
    milvus = app_milvus

    # 删除旧的 collection 如果存在
    if milvus.has_collection('test01'):
        status = milvus.drop_collection('test01')
        print(status)
        sleep(5)  # 等待5秒，确保 collection 被删除

    # 在 pymilvus 2.x 中创建集合时需要指定字段
    fields = {
        "fields": [
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 512}},
            {"name": "id", "type": DataType.INT64, "is_primary": True}
        ],
        "auto_id": False  # 根据需要选择是否自动生成ID
    }

    # 正确的调用方式
    milvus.create_collection('test01', fields)

    return milvus

def create_index(collection_name, field_name):
    # 建立连接
    connections.connect()

    # 获取集合的引用
    collection = Collection(collection_name)

    # 定义索引参数
    index_params = {
        "metric_type": "L2",  # 使用的度量类型
        "index_type": "IVF_FLAT",  # 索引类型
        "params": {"nlist": 16384}  # 索引构建参数
    }

    # 为集合中的指定字段创建索引
    index = Index(collection, field_name, index_params)
    collection.create_index(field_name=field_name, index_params=index_params)
    print(f"Index for {field_name} in {collection_name} created successfully.")

# 验证图片的完整性
def IsValidImage(pathfile):
    bValid = True
    try:
        Im.open(pathfile).verify()
    except:
        bValid = False
    return bValid

def main():
    with app.app_context():
        begin_time = time()
        vgg = VGGNet()
        milvus = create_milvus()
        db.session.query(Image).delete()  # 清空表
        db.session.commit()


        url = r'D:\python_projects\cats_and_dogs_small\train' # TODO 這邊之後要改成資料庫的路徑
        vectors, ids = [], []
        cnt = 1
        for root, dirs, files in os.walk(url):
            for file in files:
                path = os.path.join(root, file)
                if IsValidImage(path):
                    print("----", path)
                    vector = vgg.extract_feat(path)
                    vectors.append(vector)
                    image = Image(cnt, path)
                    db.session.add(image)
                    ids.append(cnt)
                    cnt += 1

        # 注意：在插入数据时需要确保数据格式和字段对应
        # 将向量和 ID 组合成符合 Milvus 2.x 要求的格式
        entities = [
            {"name": "embedding", "values": vectors, "type": DataType.FLOAT_VECTOR},
            {"name": "id", "values": ids, "type": DataType.INT64}
        ]
        
        # 使用 Milvus 2.x 的插入方法
        milvus.insert(collection_name='test01', entities=entities)

        # 刷新集合以确保数据已写入
        milvus.flush(['test01'])

        # 创建索引
        create_index('test01', 'embedding')


        db.session.commit()

        end_time = time()
        run_time = end_time - begin_time
        print('該循環程序運行時間：', run_time)

if __name__ == '__main__':
    main()
'''
import os
from time import *

import numpy as np

from app import db, Image, app_milvus
from search.vgg import VGGNet
from pymilvus import Milvus, MetricType
from PIL import Image as Im


def create_milvus():
    milvus = app_milvus
    # # 删除
    if(milvus.has_collection('test01')[1]):
        status=milvus.drop_collection(collection_name='test01')
        print(status)
        sleep(5)#等待5s，等待删除完毕

    # 创建 collection 名为 test01， 1*512， 自动创建索引的数据文件大小为 1024 MB，距离度量方式为欧氏距离（L2）的 collection 。
    param = {'collection_name': 'test01', 'dimension': 512, 'index_file_size': 1024, 'metric_type': MetricType.L2}
    milvus.create_collection(param)
    return milvus

#验证图片的完整性
def IsValidImage(pathfile):
    bValid = True
    try:
        Im.open(pathfile).verify()
    except:
        bValid = False
    return bValid


def main():
    begin_time = time()
    vgg = VGGNet()
    milvus=create_milvus()
    db.session.query(Image).delete()#情况表
    db.session.commit()

    url=r'D:\python_projects\cats_and_dogs_small\train'
    vectors,ids=[],[]
    cnt=1;
    for root,dirs,files in os.walk(url):

        for file in files:
            #获取文件路径
            path=os.path.join(root,file)
            if(IsValidImage(path)):
                print("----",path)
                vector=vgg.extract_feat(path)
                vectors.append(vector)
                image=Image(cnt,path)
                db.session.add(image)
                ids.append(cnt)
                cnt = cnt + 1
    milvus.insert(collection_name='test01', records=np.array(vectors),ids=ids)
    milvus.flush(collection_name_array=['test01'])

    db.session.commit()

    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time)  # 该循环程序运行时间： 1.4201874732

if __name__ == '__main__':
    main()
'''