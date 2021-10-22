import torch
from collections import OrderedDict


def for_detector(checkpoint):
    state_dict = checkpoint['state_dict']
    pre_dict = dict()
    pre_dict['state_dict'] = OrderedDict()
    pre_state_dict = pre_dict['state_dict']
    for k, v in state_dict.items():
        if 'detector' in k:
            new_key = k[9:]
            if 'conv2____' in new_key:
                items = ['conv2_tl', 'conv2_tr', 'conv2_bl', 'conv2_br']
                for item in items:
                    pre_state_dict[new_key.replace('conv2', item)] = state_dict[k]
            else:
                pre_state_dict[new_key] = state_dict[k]
    return pre_dict


def for_cleaner(checkpoint):
    state_dict = checkpoint['state_dict']
    pre_dict = dict()
    pre_dict['state_dict'] = OrderedDict()
    pre_state_dict = pre_dict['state_dict']
    for k, v in state_dict.items():
        if 'detector.backbone' in k:
            new_key = k[18:]
            pre_state_dict[new_key] = state_dict[k]
    return pre_dict


def for_dvd_net_cleaner(checkpoint):
    pre_dict = dict()
    pre_dict['state_dict'] = OrderedDict()
    pre_state_dict = pre_dict['state_dict']
    for k, v in checkpoint.items():
        new_key = k
        pre_state_dict[new_key] = checkpoint[k]
    return pre_dict


if __name__ == '__main__':
    # path_in = 'checkpoints/base_det/llvod_l1234_vid_a7s3.pth'
    # path_out = 'checkpoints/detector/base_resblock_selsa_tra_vid_e7_fix.pth'
    # # path_out = 'checkpoints/cleaner/base_resblock_selsa_tra_vid_e7_fix.pth'

    #
    # checkpoint = torch.load(path_in)
    # if 'detector' in path_out:
    #     save_dict = for_detector(checkpoint)
    # elif 'cleaner' in path_out:
    #     save_dict = for_cleaner(checkpoint)
    # else:
    #     raise NotImplemented
    # print('******************* finishing *******************')
    # print(save_dict)
    # torch.save(save_dict, path_out)

    path_in = 'checkpoints/fastdvdnet.pkl'
    path_out = 'checkpoints/FastDVDNet_cleaner.pth'

    checkpoints = torch.load(path_in)
    save_dict = for_dvd_net_cleaner(checkpoints)
    torch.save(save_dict, path_out)
    print('####')
