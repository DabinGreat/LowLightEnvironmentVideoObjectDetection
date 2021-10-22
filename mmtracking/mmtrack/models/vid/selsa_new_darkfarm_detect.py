import torch
import mmcv
import cv2
import numpy as np
from addict import Dict
from mmcv.image import imread, imwrite
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODELS, build_detector, build_cleaner, build_aggregator
from .base import BaseVideoDetector
# from mmtrack.core.utils import general_clean_noise_pairs


@MODELS.register_module()
class SelsaNewDarkfarmDetect(BaseVideoDetector):
    """Sequence Level Semantics Aggregation for Video Object Detection.

    This video object detector is the implementation of `SELSA
    <https://arxiv.org/abs/1907.06390>`_.
    """

    def __init__(self,
                 detector,
                 cleaner,
                 aggregator,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_type=None,
                 ):
        super(SelsaNewDarkfarmDetect, self).__init__()
        self.detector = build_detector(detector)
        self.cleaner = build_cleaner(cleaner)
        self.aggregator = build_aggregator(aggregator)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_type = loss_type
        # self.loss_stage_ratio = nn.Parameter(torch.tensor([1., 1., 1., 1.]))

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        """Initialize the weights of modules in video object detector.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_motion and pretrain.get('motion', False):
            self.init_module('motion', pretrain.get('motion', None))
        if self.with_cleaner and pretrain.get('cleaner', False):
            self.init_module('cleaner', pretrain['cleaner'])
        if self.with_aggregator and pretrain.get('aggregator', False):
            self.init_module('aggregator', pretrain.get('aggregator', None))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            ref_img (Tensor): of shape (N, 2, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                2 denotes there is two reference images for each input image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each reference image
                with shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0, and denotes the id of reference image for each
                key image.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each reference
                box with shape (num_all_ref_gts, 2) in
                [ref_img_id, class_indice].

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals (None | Tensor) : override rpn proposals with custom
                proposals. Use when `with_rpn` is False.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bboxes of reference images.

            ref_gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes of reference images can be ignored when computing the
                loss.

            ref_gt_masks (None | Tensor) : True segmentation masks for each
                box of reference image used if the architecture supports a
                segmentation task.

            ref_proposals (None | Tensor) : override rpn proposals with custom
                proposals of reference images. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(img) == 1, \
            'selsa video detector only supports 1 batch size per gpu for now.'
        isRAW = img.shape[1] % 4 == 0
        if isRAW:
            noise_img, clean_img = img[:, :4, ...], img[:, 4:, ...]
            noise_ref, clean_ref = ref_img[:, :, :4, ...], ref_img[:, :, 4:, ...]
        else:
            noise_img, clean_img = img[:, :3, ...], img[:, 3:, ...]
            noise_ref, clean_ref = ref_img[:, :, :3, ...], ref_img[:, :, 3:, ...]
        clean_imgs = torch.cat([clean_img, clean_ref[0]], dim=0)
        noise_imgs = torch.cat([noise_img, noise_ref[0]], dim=0)

        x_noise, all_x = self.detector.extract_feat(noise_imgs)
        x_clean = self.cleaner(clean_imgs)
        _x_noise, _all_x = self.aggregator(x_noise, all_x)
        x = []
        ref_x = []
        for i in range(len(_all_x)):
            x.append(_all_x[i][[0]])
            ref_x.append(_all_x[i][1:])

        losses = dict()

        # clean and noise mse loss
        if self.with_cleaner:
            feats_losses = dict()
            if self.loss_type == 'l1':
                feat_loss = nn.L1Loss()
            elif self.loss_type == 'l2':
                feat_loss = nn.MSELoss()
            elif self.loss_type == 'smooth_l1':
                feat_loss = nn.SmoothL1Loss()
            else:
                raise NotImplementedError()

            # ratio = len(x_noise) * F.softmax(self.loss_stage_ratio, dim=0)
            for i in range(len(x_noise)):
                # feats_losses[f'loss_{self.loss_type}_{i}'] = feat_loss(x_noise[i], x_clean[i]) * ratio[i]
                # feats_losses[f'stage_ratio_para_{i}'] = ratio[i]
                feats_losses[f'loss_{self.loss_type}_{i}_u'] = feat_loss(x_noise[i], x_clean[i])
                feats_losses[f'loss_{self.loss_type}_{i}_d'] = feat_loss(_x_noise[i], x_clean[i])
            losses.update(feats_losses)

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas[0])
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (Tensor | None): of shape (1, N, C, H, W) encoding input
                reference images. Typically these should be mean centered and
                std scaled. N denotes the number of reference images. There
                may be no reference images in some cases.

            ref_img_metas (list[list[dict]] | None): The first list only has
                one element. The second list contains image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

        Returns:
            tuple(x, img_metas, ref_x, ref_img_metas): x is the multi level
                feature maps of `img`, ref_x is the multi level feature maps
                of `ref_img`.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img_metas = ref_img_metas[0]
                ref_noise_x, ref_x = self.detector.extract_feat(ref_img[0])
                self.memo.feats_ref_x = []
                self.memo.feats_ref_noise_x = []
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                for i in range(len(ref_x)):
                    self.memo.feats_ref_x.append(ref_x[i])
                for i in range(len(ref_noise_x)):
                    self.memo.feats_ref_noise_x.append(ref_noise_x[i])

            noise_x, x = self.detector.extract_feat(img)
            ref_noise_x = self.memo.feats_ref_noise_x.copy()
            ref_x = self.memo.feats_ref_x.copy()
            for i in range(len(x)):
                ref_x[i] = torch.cat([ref_x[i], x[i]], dim=0)
            for i in range(len(noise_x)):
                ref_noise_x[i] = torch.cat([ref_noise_x[i], noise_x[i]], dim=0)
            ref_noise_x, ref_x = self.aggregator(ref_noise_x, ref_x)
            x = []
            for i in range(len(ref_x)):
                x.append(ref_x[i][[-1]])
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas.extend(img_metas)

        # test with fixed stride
        else:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img = ref_img[0]
                self.memo.img_metas = ref_img_metas[0]
                _, ref_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
                    x.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x = []
                ref_imgs = torch.cat([self.memo.img, ref_img[0]], dim=0)[1:]
                _, ref_x = self.detector.extract_feat(ref_imgs)
                for i in range(len(ref_x)):
                    self.memo.feats[i] = ref_x[i]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img_metas.extend(ref_img_metas[0])
                self.memo.img_metas = self.memo.img_metas[1:]
                self.memo.img = ref_imgs
            else:
                assert ref_img is None
                _, x = self.detector.extract_feat(ref_img[0])

            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i][num_left_ref_imgs] = x[i]
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas[num_left_ref_imgs] = img_metas[0]

        return x, img_metas, ref_x, ref_img_metas

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (list[Tensor] | None): The list only contains one Tensor
                of shape (1, N, C, H, W) encoding input reference images.
                Typically these should be mean centered and std scaled. N
                denotes the number for reference images. There may be no
                reference images in some cases.

            ref_img_metas (list[list[list[dict]]] | None): The first and
                second list only has one element. The third list contains
                image information dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain 'filename',
                'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
                the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

            proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

            rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """
        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]
        x, img_metas, ref_x, ref_img_metas = self.extract_feats(
            img, img_metas, ref_img, ref_img_metas)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        outs = self.detector.roi_head.simple_test(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            img_metas,
            rescale=rescale)

        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def imshow(self, img, win_name='', wait_time=0):
        """Show an image.

        Args:
            img (str or ndarray): The image to be displayed.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
        """
        cv2.imshow(win_name, imread(img))
        if wait_time == 0:  # prevent from hanging if windows was closed
            while True:
                ret = cv2.waitKey(1)

                closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                # if user closed window or if some key pressed
                if closed or ret != -1:
                    break
        else:
            ret = cv2.waitKey(wait_time)

    def imshow_det_bboxes(self,
                          img,
                          bboxes,
                          labels,
                          class_names=None,
                          score_thr=0,
                          bbox_color=None,
                          text_color=None,
                          thickness=3,
                          font_scale=0.5,
                          show=True,
                          win_name='',
                          wait_time=0,
                          out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        assert len(class_names) == 8

        class_clolor = {'person': (255, 105, 180), 'cow': (60, 179, 113),
                        'sheep': (100, 149, 237), 'dog': (255, 165, 0),
                        'rabbit': (72, 209, 204), 'cat': (153, 50, 204),
                        'hen': (255, 245, 238), 'duck': (105, 105, 105)}

        img = imread(img)
        img = np.ascontiguousarray(img)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            color = class_clolor[class_names[label]]
            tl = round(0.004 * (img.shape[0] + img.shape[1]) / 2) + 1
            tf = max(tl - 1, 1)
            cv2.rectangle(
                img, left_top, right_bottom, color, thickness=tl, lineType=cv2.LINE_AA)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f' {bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        if show:
            self.imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=None,
                    text_color=None,
                    thickness=5,
                    font_scale=5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # TODO: make it support tracking
        img = mmcv.imread(img)
        img = img.copy()
        assert isinstance(result, dict)
        bbox_result = result.get('bbox_results', None)
        segm_result = result.get('segm_results', None)
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
