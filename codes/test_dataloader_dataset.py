import os.path as osp
import logging
import math
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util

# root path of EDVR repo
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
logging.getLogger().setLevel(logging.DEBUG)
logging.info(root)


#####################
## REDS
#####################
def RED_opts():
  logging.info('Testing REDS dataset')
  opt={}
  opt['name'] = 'test_REDS'
  opt['dataroot_GT'] = osp.join(root,'datasets/REDS/train/sharp_wval.lmdb')
  opt['dataroot_LQ'] = osp.join(root,'datasets/REDS/train/sharp_bicubic_wval.lmdb')
  logging.info(f"GT={opt['dataroot_GT']}")
  logging.info(f"LQ={opt['dataroot_LQ']}")
  opt['mode'] = 'REDS'
  opt['N_frames'] = 5
  opt['phase'] = 'train'
  opt['use_shuffle'] = True
  opt['n_workers'] = 8
  opt['batch_size'] = 16
  opt['GT_size'] = 256
  opt['LQ_size'] = 64
  opt['scale'] = 4
  opt['use_flip'] = True
  opt['use_rot'] = True
  opt['interval_list'] = [1]
  opt['random_reverse'] = False
  opt['border_mode'] = False
  opt['cache_keys'] = 'REDS_trainval_keys.pkl'
  return opt

#####################
## Vimeo90K
#####################
def Vimeo90k_opts():
  logging.info('Testing Vimeo90k dataset')
  opt={}
  opt['name'] = 'test_Vimeo90K'
  opt['dataroot_GT'] = osp.join(root,'datasets/vimeo90k/vimeo90k_train_GT.lmdb')
  opt['dataroot_LQ'] = osp.join(root,'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb')
  opt['mode'] = 'Vimeo90K'
  opt['N_frames'] = 7
  opt['phase'] = 'train'
  opt['use_shuffle'] = True
  opt['n_workers'] = 8
  opt['batch_size'] = 16
  opt['GT_size'] = 256
  opt['LQ_size'] = 64
  opt['scale'] = 4
  opt['use_flip'] = True
  opt['use_rot'] = True
  opt['interval_list'] = [1]
  opt['random_reverse'] = False
  opt['border_mode'] = False
  opt['cache_keys'] = 'Vimeo90K_train_keys.pkl'
  return opt
###############################################################################
opt = RED_opts() # or Vimeo90k_opts()
opt['data_type'] = 'lmdb'  # img | lmdb | mc
opt['dist'] = False
opt['gpu_ids'] = [0]

util.mkdir('tmp')
train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt, opt, None)
nrow = int(math.sqrt(opt['batch_size']))
if opt['phase'] == 'train':
    padding = 2
else:
    padding = 0

logging.info('start...')
for i, data in enumerate(train_loader):
    if i > 5:
        break
    logging.info(i)
    LQs = data['LQs']
    GT = data['GT']
    key = data['key']

    # save LQ images
    for j in range(LQs.size(1)):
        torchvision.utils.save_image(LQs[:, j, :, :, :], 'tmp/LQ_{:03d}_{}.png'.format(i, j),
                                     nrow=nrow, padding=padding, normalize=False)
    torchvision.utils.save_image(GT, 'tmp/GT_{:03d}.png'.format(i), nrow=nrow, padding=padding,
                                 normalize=False)
