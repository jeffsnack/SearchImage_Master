import os
import shutil

base_dir = r'D:\python_projects\cats_and_dogs_small'

data_dir = r'D:\python_projects\dogs-vs-cats-redux-kernels-edition\train'

train_dir = os.path.join(base_dir,'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
#os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,'cats')
#os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
#os.mkdir(train_dogs_dir)


validation_cats_dir = os.path.join(validation_dir,'cats')
#os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dog')
#os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
#os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')
#os.mkdir(test_dogs_dir)

names = ['cat.{}.jpg'.format(i) for i in range(8000)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(train_cats_dir, na)
    shutil.copyfile(src, dst)

names = ['cat.{}.jpg'.format(i) for i in range(8000, 10500)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(validation_cats_dir, na)
    shutil.copyfile(src, dst)

names = ['cat.{}.jpg'.format(i) for i in range(10500, 12500)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(test_cats_dir, na)
    shutil.copyfile(src, dst)

names = ['dog.{}.jpg'.format(i) for i in range(8000)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(train_dogs_dir, na)
    shutil.copyfile(src, dst)

names = ['dog.{}.jpg'.format(i) for i in range(8000, 10500)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(validation_dogs_dir, na)
    shutil.copyfile(src, dst)

names = ['dog.{}.jpg'.format(i) for i in range(10500, 12500)]

for na in names:
    src = os.path.join(data_dir, na)
    dst = os.path.join(test_dogs_dir, na)
    shutil.copyfile(src, dst)


