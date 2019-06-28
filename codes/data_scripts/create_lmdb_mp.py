'''
create lmdb files for Vimeo90K / REDS training dataset (multiprocessing)

[mbs] modification so to avoid filling up memory
[mbs] modification to generate vimeo90k LR with OpenCV bicubic
'''

import sys
import os.path as osp
import os
import glob
import pickle
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np
import lmdb
import cv2
from tqdm import tqdm

# root path of EDVR repo
root = osp.dirname(osp.dirname(osp.abspath(__file__)))

try:
    sys.path.append(root)
    import data.util as data_util
    import utils.util as util
except ImportError:
    pass

root=osp.dirname(root)

def create_vimeo90k_LR(scale_factor=4):
  from scipy.misc import imresize
  img_folder_GT = osp.join(root,'datasets/vimeo90k/vimeo_septuplet/sequences')
  img_folder_LR = osp.join(root,'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences')
  txt_file = osp.join(root,'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt')
  
  #### read all the image paths to a list
  print('Reading image path list ...')
  with open(txt_file) as f:
      train_l = f.readlines()
      train_l = [v.strip() for v in train_l]
  all_img_list = []

  for line in train_l:
      folder = line.split('/')[0]
      sub_folder = line.split('/')[1]
      file_l = glob.glob(osp.join(img_folder_GT, folder, sub_folder) + '/*')
      all_img_list.extend(file_l)

  all_img_list = sorted(all_img_list)
  nimgs=len(all_img_list)

  ### create LR directories
  print('Creating LR directories ...')
  for img_path in all_img_list:
    file_lr=osp.join(img_folder_LR, img_path[len(img_folder_GT)+1:])
    directory=osp.dirname(file_lr)
    if not osp.exists(directory):
      os.makedirs(directory)

  ### convert images
  pbar = tqdm(total=nimgs)
  def convert_image(img_path):
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    lr_h = frame.shape[0] // scale_factor
    lr_w = frame.shape[1] // scale_factor

#    frame_lr = imresize(frame, (lr_h, lr_w), interp='bicubic') # scikit is on int, loose precision
    frame_lr=cv2.resize(frame, (lr_h, lr_w), cv2.INTER_CUBIC);
    file_lr=osp.join(img_folder_LR, img_path[len(img_folder_GT)+1:])
#    print(f'Writing {lr_h}x{lr_w} as {file_lr}')
        
    cv2.imwrite(file_lr, frame_lr)
    pbar.update()
    
  with ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
    executor.map(convert_image,all_img_list)

  pbar.close()  

def vimeo90k(mode='GT',overwrite=True):
    '''create lmdb for the Vimeo90K dataset, each image with fixed size
    GT: [3, 256, 448] Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]  With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
        
    mode='GT' or 'LR'
    '''
    #### configurations
    print(f'**** Creating vimeo90k lmdb {mode} database')
    if mode == 'GT':
        img_folder = osp.join(root,'datasets/vimeo90k/vimeo_septuplet/sequences')
        lmdb_save_path = osp.join(root,'datasets/vimeo90k/vimeo90k_train_GT.lmdb')
        txt_file = osp.join(root,'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt')
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        img_folder = osp.join(root,'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences')
        lmdb_save_path = osp.join(root,'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb')
        txt_file = osp.join(root,'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt')
        H_dst, W_dst = 64, 112

    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if not overwrite and osp.exists(lmdb_save_path):
        print(f'Folder [{lmdb_save_path}] already exists. Exit...')
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]

    all_img_list = []
    keys=[]

    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append(f'{folder}_{sub_folder}_{j+1}')

    all_img_list = sorted(all_img_list)
    keys=sorted(keys)

    if mode == 'GT':  # read the 4th frame only for GT mode
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    nimgs=len(all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing ....')
    img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED)
    data_size_per_img = img.nbytes
    print('data size per image is: ', data_size_per_img)
    
    pbar = tqdm(total=nimgs,unit='files')

    def path2key(img_path):
      split_rlt = img_path.split('/')
      folder = split_rlt[-3]
      subfolder = split_rlt[-2]
      file = split_rlt[-1].split('.png')[0]
      file=file[len('img')-1:]
      return folder, subfolder, file

    def read_image(img_path):
      folder, subfolder, file=path2key(img_path)
      key=f'{folder}_{subfolder}_{file}'
      pbar.set_postfix(file=key, refresh=False)
      pbar.update()
      data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

      #idx=int(a)*100+int(b)
      #print(f'Read [{idx}] {key_byte} = {img_path} data sz={len(data)}')
      H, W, C = data.shape  # fixed shape
      assert H == H_dst and W == W_dst and C == 3, f'different shape {H}x{W}x{C} should be {H_dst}x{W_dst}x3.'

      return key, data
    
    item_id=1
    batch_size=1024
    env = lmdb.open(lmdb_save_path, map_size=data_size_per_img*nimgs*10)
    txn=env.begin(write=True)
    for img_path in all_img_list:
      item_id+=1
      key,data=read_image(img_path)
      txn.put(key.encode('ascii'), data)
      
      # write batch
      if(item_id + 1) % batch_size == 0:
          txn.commit()
          txn = env.begin(write=True)
    
    # write last batch
    if (item_id+1) % batch_size != 0:
      txn.commit()

    env.close()
    pbar.close()  
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'Vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo90K_train_LR'
    meta_info['resolution'] = f'3_{H_dst}_{W_dst}'
    
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add(f'{a}_{b}')
        
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def REDS(mode = 'train_sharp',overwrite=True):
    '''create lmdb for the REDS dataset, each image with fixed size
    GT: [3, 720, 1280], key: 000_00000000
    LR: [3, 180, 320], key: 000_00000000

    for each dataset
      keys follow the format <sequence id>_<image idx>
    
    mode = train_sharp | train_sharp_bicubic | train_blur_bicubic| train_blur | train_blur_comp
    '''
    #### configurations   
    print(f'**** Creating REDS lmdb database for {mode} ****')
    if mode == 'train_sharp':
        img_folder = osp.join(root,'datasets/REDS/train/sharp')
        lmdb_save_path = osp.join(root,'datasets/REDS/train/sharp_wval.lmdb')
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_bicubic':
        img_folder = osp.join(root,'datasets/REDS/train/sharp_bicubic')
        lmdb_save_path = osp.join(root,'datasets/REDS/train/sharp_bicubic_wval.lmdb')
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur_bicubic':
        img_folder = osp.join(root,'datasets/REDS/train/blur_bicubic')
        lmdb_save_path = osp.join(root,'datasets/REDS/train/blur_bicubic_wval.lmdb')
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur':
        img_folder = osp.join(root,'datasets/REDS/train/blur')
        lmdb_save_path = osp.join(root,'datasets/REDS/train/blur_wval.lmdb')
        H_dst, W_dst = 720, 1280
    elif mode == 'train_blur_comp':
        img_folder = osp.join(root,'datasets/REDS/train/blur_comp')
        lmdb_save_path = osp.join(root,'datasets/REDS/train/blur_comp_wval.lmdb')
        H_dst, W_dst = 720, 1280
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if not overwrite and osp.exists(lmdb_save_path):
        print(f'Folder [{lmdb_save_path}] already exists. Exit...')
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(img_folder)
    nimgs=len(all_img_list)
    print(f'Total number of images: {nimgs} ...')    

    #### create lmdb environment
    print('Write lmdb...')
    img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED)
    data_size_per_img = img.nbytes
    print('data size per image is: ', data_size_per_img)

    #### write data to lmdb
    pbar = tqdm(total=nimgs,unit='files')
    
    keys = set()
    def path2key(img_path):
      split_rlt = img_path.split('/')
      a = split_rlt[-2]
      b = split_rlt[-1].split('.png')[0]
      return f'{a}_{b}'

    def read_image(img_path):
      key=path2key(img_path)
      keys.add(key)
      pbar.set_postfix(file=key, refresh=False)
      pbar.update()
      data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

      if 'flow' in mode:
        H, W = data.shape
        assert H == H_dst and W == W_dst, 'different shape.'
      else:
        H, W, C = data.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'

      with env.begin(write=True) as txn:
        txn.put(key.encode('ascii'), data)

    env = lmdb.open(lmdb_save_path, map_size=data_size_per_img*nimgs*10)
    with ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
      executor.map(read_image,all_img_list)
       
    env.close()
    pbar.close()  
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = f'REDS_{mode}_wval'
    if 'flow' in mode:
        meta_info['resolution'] = f'1_{H_dst}_{W_dst}'
    else:
        meta_info['resolution'] = f'3_{H_dst}_{W_dst}'
        
    keys=sorted(keys)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def test_lmdb(dataroot, dataset='REDS'):
  env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
  meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
  print('Name: ', meta_info['name'])
  print('Resolution: ', meta_info['resolution'])
  print('# keys: ', len(meta_info['keys']))
  # read one image
  if dataset == 'vimeo90k':
      key = '00001_0001_4'
  else:
      key = '000_00000000'
  print(f'Reading {key} for test.')
  with env.begin(write=False) as txn:
      buf = txn.get(key.encode('ascii'))
  img_flat = np.frombuffer(buf, dtype=np.uint8)
  C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
  img = img_flat.reshape(H, W, C)
  cv2.imwrite('test.png', img)


if __name__ == "__main__":
#  vimeo90k('GT')
#  
#  create_vimeo90k_LR()      
  vimeo90k('LR')
  test_lmdb(osp.join(root,'datasets/vimeo90k/vimeo90k_train_GT.lmdb'), 'vimeo90k')
  
#  modes=['train_sharp','train_sharp_bicubic','train_blur','train_blur_bicubic','train_blur_comp']
#  for mode in modes:
#    REDS(mode) 
#  
#  test_lmdb(osp.join(root,'datasets/REDS/train/sharp_wval.lmdb'), 'REDS')
