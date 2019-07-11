'''
test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
write to txt log file
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.modules.EDVR_arch as EDVR_arch

# root path of EDVR repo
root = osp.dirname(osp.abspath(__file__))


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'sharp_bicubic'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    ############################################################################
    #### model
    if data_mode == 'Vid4':
        if stage == 1:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth')
        else:
            raise ValueError('Vid4 does not support stage 2.')
    elif data_mode == 'sharp_bicubic':
        if stage == 1:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_SR_L.pth')
        else:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth')
    elif data_mode == 'blur_bicubic':
        if stage == 1:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth')
        else:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth')
    elif data_mode == 'blur':
        if stage == 1:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth')
        else:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth')
    elif data_mode == 'blur_comp':
        if stage == 1:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth')
        else:
            model_path = osp.join(root, '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth')
    else:
        raise NotImplementedError
    if data_mode == 'Vid4':
        N_in = 7  # use N_in images to restore one HR image
    else:
        N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 40
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True
    if stage == 2:
        HR_in = True
        back_RBs = 20
    model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = osp.join(root, '../datasets/Vid4/BIx4/*')
        GT_dataset_folder = osp.join(root, '../datasets/Vid4/GT/*')
    else:
        if stage == 1:
            test_dataset_folder = osp.join(root, f'../datasets/REDS4/{data_mode}/*')
        else:
            raise ValueError('You should modify the test_dataset_folder path for stage 2')
        GT_dataset_folder = osp.join(root, '../datasets/REDS4/GT/*')

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    device = torch.device('cuda')
    save_folder = f'../results/{data_mode}'
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info(f'Data: {data_mode} - {test_dataset_folder}')
    logger.info(f'Padding mode: {padding}')
    logger.info(f'Model path: {model_path}')
    logger.info(f'Save images: {save_imgs}')
    logger.info(f'Flip Test: {flip_test}')

    def read_image(img_path):
        '''read one image from img_path
        Return img: HWC, BGR, [0,1], numpy
        '''
        img_GT = cv2.imread(img_path)
        img = img_GT.astype(np.float32) / 255.
        return img

    def read_seq_imgs(img_seq_path):
        '''read a sequence of images'''
        img_path_l = sorted(glob.glob(img_seq_path + '/*'))
        img_l = [read_image(v) for v in img_path_l]
        # stack to TCHW, RGB, [0,1], torch
        imgs = np.stack(img_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        return imgs

    def index_generation(crt_i, max_n, N, padding='reflection'):
        '''
        padding: replicate | reflection | new_info | circle
        '''
        max_n = max_n - 1
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = N + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i > max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - N
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l

    def single_forward(model, imgs_in):
        with torch.no_grad():
            model_output = model(imgs_in)
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(test_dataset_folder))
    sub_folder_GT_l = sorted(glob.glob(GT_dataset_folder))
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    sub_folder_name_l = []

    # for each sub-folder
    for sub_folder, sub_folder_GT in zip(sub_folder_l, sub_folder_GT_l):
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        img_path_l = sorted(glob.glob(sub_folder + '/*'))
        max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT, '*'))):
            img_GT_l.append(read_image(img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center = 0, 0, 0
        cal_n_border, cal_n_center = 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            c_idx = int(osp.splitext(osp.basename(img_path))[0])
            select_idx = index_generation(c_idx, max_idx, N_in, padding=padding)
            # get input images
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            output = single_forward(model, imgs_in)
            output_f = output.data.float().cpu().squeeze(0)

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output

                output_f = output_f / 4

            output = util.tensor2img(output_f)

            # save imgs
            if save_imgs:
                cv2.imwrite(osp.join(save_sub_folder, f'{c_idx:08d}.png'), output)

            #### calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on Y channels
            if data_mode == 'Vid4':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT)
                output = data_util.bgr2ycbcr(output)
            if crop_border == 0:
                cropped_output = output
                cropped_GT = GT
            else:
                cropped_output = output[crop_border:-crop_border, crop_border:-crop_border]
                cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border]
            crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
            logger.info(f'{img_idx+1:3d} - {c_idx:25}.png \tPSNR: {crt_psnr:.6f} dB')

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                cal_n_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                cal_n_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (cal_n_center + cal_n_border)
        avg_psnr_center = avg_psnr_center / cal_n_center
        if cal_n_border == 0:
            avg_psnr_border = 0
        else:
            avg_psnr_border = avg_psnr_border / cal_n_border

        logger.info(f'Folder {sub_folder_name} - Average PSNR: {avg_psnr:.6f} dB for {(cal_n_center + cal_n_border)} frames; '
                    f'Center PSNR: {avg_psnr_center:.6f} dB for {cal_n_center} frames; '
                    f'Border PSNR: {avg_psnr_border:.6f} dB for {cal_n_border} frames.')

        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

    logger.info('################ Tidy Outputs ################')
    for name, psnr, psnr_center, psnr_border in zip(sub_folder_name_l, avg_psnr_l,
                                                    avg_psnr_center_l, avg_psnr_border_l):
        logger.info(f'Folder {name} - Average PSNR: {psnr:.6f} dB. '
                    f'Center PSNR: {psnr_center:.6f} dB. '
                    f'Border PSNR: {psnr_border:.6f} dB.')
    logger.info('################ Final Results ################')
    logger.info(f'Data: {data_mode} - {test_dataset_folder}')
    logger.info(f'Padding mode: {padding}')
    logger.info(f'Model path: {model_path}')
    logger.info(f'Save images: {save_imgs}')
    logger.info(f'Flip Test: {flip_test}')
    logger.info(f'Total Average PSNR: {sum(avg_psnr_l) / len(avg_psnr_l):.6f} dB for {len(sub_folder_l)} clips. '
                f'Center PSNR: {sum(avg_psnr_center_l) / len(avg_psnr_center_l):.6f} dB. '
                f'Border PSNR: {sum(avg_psnr_border_l) / len(avg_psnr_border_l):.6f} dB.')


if __name__ == '__main__':
    main()
