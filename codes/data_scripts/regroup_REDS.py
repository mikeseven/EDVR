'''
put REDS validatation sets with train sets

[mbs] modification: do it for all REDS sets
'''

import os
import glob
import os.path as osp
from tqdm import tqdm
root = osp.dirname(osp.dirname(osp.abspath(__file__)))

sets=['sharp','sharp_bicubic/X4','blur','blur_bicubic/X4','blur_comp']
train_path = os.path.join(root,'../datasets/REDS/train')
val_path = os.path.join(root,'../datasets/REDS/val')

# mv the val set
for zset in tqdm(sets):
  val_folders = glob.glob(os.path.join(val_path,zset, '*'))
  for folder in tqdm(val_folders):
      new_folder_idx = '{:03d}'.format(int(folder.split('/')[-1]) + 240)
#      print('cp -r {} {}'.format(folder, os.path.join(train_path, zset, new_folder_idx)))
      os.system('cp -r {} {}'.format(folder, os.path.join(train_path, zset, new_folder_idx)))
