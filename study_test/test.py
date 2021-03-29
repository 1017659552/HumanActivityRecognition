import os
dir = 'D:\SWUFEthesis\data\KTH_preprocess_v5_reorder'
for split in os.listdir(dir):
    print(split)
    count = 0
    split_dir = os.path.join(dir,split)
    for cls in os.listdir(split_dir):
        for item in os.listdir(os.path.join(split_dir,cls)):
            frame_num = os.listdir(os.path.join(split_dir,cls,item))
            count += len(frame_num)
    print(count)
