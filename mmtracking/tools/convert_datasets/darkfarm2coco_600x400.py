import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import defaultdict

import mmcv
from tqdm import tqdm


CLASSES = ('person', 'cow', 'sheep', 'dog',
           'rabbit', 'cat', 'hen', 'duck')

cats_id_maps = {}
for k, v in enumerate(CLASSES, 1):
    cats_id_maps[v] = k

def parse_args():
    parser = argparse.ArgumentParser(
        description='DARK FARM to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        default='/data/DarkFarm2',
        help='root directory of DARK FARM annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='/data/DarkFarm2/Annotations',
        help='directory to save coco formatted label file',
    )
    return parser.parse_args()


def parse_train_list(ann_dir):
    img_list = osp.join(ann_dir, 'Lists/darkfarm_vid_e2e_train.txt')
    img_list = mmcv.list_from_file(img_list)
    train_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        if info[0] not in train_infos:
            train_infos[info[0]] = dict(
                vid_train_frames=[int(info[1])],
                num_frames=int(info[-1]))
        else:
            train_infos[info[0]]['vid_train_frames'].append(int(info[1]))
    return train_infos


def parse_val_list(ann_dir, prefix):
    img_list = osp.join(ann_dir, 'Lists/%s.txt' % prefix)
    img_list = mmcv.list_from_file(img_list)
    val_infos = defaultdict(list)
    for info in img_list:
        info = info.split(' ')
        val_infos[info[0]] = dict(num_frames=int(info[-1]))
    return val_infos


def convert_darkfarm(VID, ann_dir, save_dir, mode='train', prefix=''):
    assert mode in ['train', 'val']
    records = dict(
        vid_id=1,
        img_id=1,
        ann_id=1,
        global_instance_id=1,
        num_vid_train_frames=0,
        num_no_objects=0)
    obj_num_classes = dict()
    if mode == 'train':
        vid_infos = parse_train_list(ann_dir)
    else:
        vid_infos = parse_val_list(ann_dir, prefix)
    for vid_info in tqdm(vid_infos):
        instance_id_maps = dict()
        vid_train_frames = vid_infos[vid_info].get('vid_train_frames', [])
        records['num_vid_train_frames'] += len(vid_train_frames)
        video = dict(
            id=records['vid_id'],
            name=vid_info,
            vid_train_frames=vid_train_frames
        )
        VID['videos'].append(video)
        num_frames = vid_infos[vid_info]['num_frames']
        for frame_id in range(num_frames):
            is_vid_train_frame = True if frame_id in vid_train_frames else False
            img_prefix = osp.join(vid_info, '%d' % frame_id)
            xml_prefix = vid_info.split('/')[0] + os.sep + vid_info.split('/')[1] \
                         + os.sep + vid_info.split('/')[2]
            xml_name = osp.join(ann_dir, xml_prefix, 'GT', '%d.xml' % frame_id)
            # parse XML annotation file
            tree = ET.parse(xml_name)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            n_width = 600
            height = int(size.find('height').text)
            n_height = 400
            image = dict(
                file_name=f'{img_prefix}.png',
                height=n_height,
                width=n_width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=records['vid_id'],
                is_vid_train_frame=is_vid_train_frame
            )
            VID['images'].append(image)
            if root.findall('object') == []:
                print(xml_name, 'has no objects.')
                records['num_no_objects'] += 1
                records['img_id'] += 1
                continue
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in CLASSES:
                    continue
                category_id = cats_id_maps[name]
                bnd_box = obj.find('bndbox')
                x1, y1, x2, y2 = [
                    max(min(int(int(bnd_box.find('xmin').text) * 600 / width), 600), 0),
                    max(min(int(int(bnd_box.find('ymin').text) * 400 / height), 400), 0),
                    max(min(int(int(bnd_box.find('xmax').text) * 600 / width), 600), 0),
                    max(min(int(int(bnd_box.find('ymax').text) * 400 / height), 400), 0)
                ]
                w = x2 - x1
                h = y2 - y1
                # track_id = obj.find('trackid').text
                # if track_id in instance_id_maps:
                #     instance_id = instance_id_maps[track_id]
                # else:
                #     instance_id = records['global_instance_id']
                #     records['global_instance_id'] += 1
                #     instance_id_maps[track_id] = instance_id
                # occluded = obj.find('occluded').text
                # generated = obj.find('generated').text
                ann = dict(
                    id=records['ann_id'],
                    video_id=records['vid_id'],
                    image_id=records['img_id'],
                    category_id=category_id,
                    instance_id=records['global_instance_id'],
                    bbox=[x1, y1, w, h],
                    area=w * h,
                    iscrowd=False,
                    occluded=False,
                    generated=False
                )
                if category_id not in obj_num_classes:
                    obj_num_classes[category_id] = 1
                else:
                    obj_num_classes[category_id] += 1
                VID['annotations'].append(ann)
                records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1
        mmcv.dump(VID, osp.join(save_dir, prefix + '.json'))
        print(f'-----ImageNet VID {mode}------')
        print(f'{records["vid_id"] - 1} videos')
        print(f'{records["img_id"] - 1} images')
        print(f'{records["num_vid_train_frames"]} train frames for video detection')
        print(f'{records["num_no_objects"]} images have no objects')
        print(f'{records["ann_id"] - 1} objects')
        print('-----------------------')
        # for i in range(1, len(CLASSES) + 1):
        #     print(f'Class {i} {CLASSES[i - 1]} has {obj_num_classes[i]} objects.')


def main():
    args = parse_args()
    categories = []
    for k, v in enumerate(CLASSES, 1):
        categories.append(
            dict(id=k, name=v)
        )

        txt_list = os.path.join(args.input, 'Lists')
        txt_to_be = ['darkfarm_vid_e2e_train.txt', 'darkfarm_vid_e2e_val.txt']
        # for txt in os.listdir(txt_list):
        #     if txt in txt_to_be:
        #         mode = 'train' if 'train' in txt else 'val'
        #         VID = defaultdict(list)
        #         VID['categories'] = categories
        #         prefix = txt[:-4]
        #         convert_darkfarm(VID, args.input, args.output, mode, prefix)
        #     else:
        #         continue
        for txt in txt_to_be:
            mode = 'train' if 'train' in txt else 'val'
            VID = defaultdict(list)
            VID['categories'] = categories
            prefix = txt[:-4]
            convert_darkfarm(VID, args.input, args.output, mode, prefix)


if __name__ == '__main__':
    main()