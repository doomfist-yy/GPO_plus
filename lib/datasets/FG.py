import os
import json
import numpy as np

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def load_npy(filename):
    return np.load(filename, allow_pickle=True)

# 加载提供的np文件和json文件
flickr_filenames = load_npy('/home/crossai/BCY/vse_infty-master/data/f30k/Flickr30K_FG_img_filenames.npy')
mscoco_filenames = load_npy('/home/crossai/BCY/vse_infty-master/data/coco/MSCOCO_FG_img_filenames.npy')
print("flickr_filenames:",flickr_filenames)
print("mscoco_filenames:",mscoco_filenames)
flickr_annotations = load_json('/home/crossai/BCY/vse_infty-master/data/f30k/Flickr30K_FG_ann.json')
mscoco_annotations = load_json('/home/crossai/BCY/vse_infty-master/data/coco/MSCOCO_FG_ann.json')

# 创建图像文件名和注释的映射
flickr_filenames_set = set(flickr_filenames)
mscoco_filenames_set = set(mscoco_filenames)

# 更新文件
def filter_dataset(dataset_folder, dataset_annotations, filenames_set):
    filtered_annotations = {}
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            if filename in filenames_set:
                print(f"Found matched filename: {filename}")
                filtered_annotations[filename] = dataset_annotations[filename]
    return filtered_annotations

# 清洗和过滤COCO数据集
coco_images_folder = '/home/crossai/BCY/vse_infty-master/data/coco/images'
coco_annotations = load_json('/home/crossai/BCY/vse_infty-master/data/coco/id_mapping.json')

filtered_coco_annotations = filter_dataset(coco_images_folder, coco_annotations, mscoco_filenames_set)

# 保存清洗后的注释文件
with open('/home/crossai/BCY/vse_infty-master/data/coco/precomp_fg/filtered_coco_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_coco_annotations, f, ensure_ascii=False, indent=4)

# caps.txt
caps_files = ['dev_caps.txt', 'test_caps.txt', 'testall_caps.txt', 'train_caps.txt']
for caps_file in caps_files:
    with open(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp', caps_file), 'r', encoding='utf-8') as f:
        data = f.readlines()
    filtered_data = [line.strip() for line in data if line.strip().split('\t')[0] in mscoco_filenames_set]
    print(f"Filtered {caps_file}: {len(filtered_data)} lines retained out of {len(data)}")
    with open(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp_fg', 'filtered_' + caps_file), 'w', encoding='utf-8') as f:
        f.writelines(filtered_data)

# ids.txt
ids_files = ['dev_ids.txt', 'test_ids.txt', 'testall_ids.txt', 'train_ids.txt']
for ids_file in ids_files:
    with open(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp', ids_file), 'r', encoding='utf-8') as f:
        data = f.readlines()
    filtered_data = [line.strip() for line in data if line.strip() in mscoco_filenames_set]
    print(f"Filtered {ids_file}: {len(filtered_data)} lines retained out of {len(data)}")
    with open(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp_fg', 'filtered_' + ids_file), 'w', encoding='utf-8') as f:
        f.writelines(filtered_data)

# ims.npy
ims_files = ['dev_ims.npy', 'test_ims.npy', 'testall_ims.npy', 'train_ims.npy']
for ims_file in ims_files:
    ims_data = load_npy(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp', ims_file))
    filtered_ims_data = [str(im) for im in ims_data if str(im).split('.')[0] in mscoco_filenames_set]
    print(f"Filtered {ims_file}: {len(filtered_ims_data)} images retained out of {len(ims_data)}")
    np.save(os.path.join('/home/crossai/BCY/vse_infty-master/data/coco/precomp_fg', 'filtered_' + ims_file), filtered_ims_data)
    
