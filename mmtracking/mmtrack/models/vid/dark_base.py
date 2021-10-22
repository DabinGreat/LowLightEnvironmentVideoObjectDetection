import torch
from addict import Dict

from mmdet.core import bbox2result
from mmtrack.core.utils import general_clean_noise_pairs, plot_noise_clean_pairs
from ..builder import MODELS, build_detector, build_aggregator
from .base import BaseVideoDetector


@MODELS.register_module()
class DarkBase(BaseVideoDetector):
    """Base class for video object detector."""

    def __init__(self,
                 detector,
                 aggregator,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DarkBase, self).__init__()
        self.detector = build_detector(detector)
        self.aggregator = build_aggregator(aggregator)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
        assert len(img) == 1, \
            'BASE video detectors only support 1 batch size per gpu for now.'

        all_imgs = torch.cat((ref_img[0], img), dim=0)
        noise, clean = general_clean_noise_pairs(all_imgs, constant=[0.5, 0.5])

        all_x = self.detector.extract_feat(noise)
        # all_x = self.detector.extract_feat(all_imgs)
        x = []
        for i in range(len(all_x)):  # index 0 is the noise features
            agg_x = self.aggregator(all_x[i][[-1]], all_x[i][:-1])
            x.append(agg_x)

        losses = dict()

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            # RPN forward and loss
            if self.detector.with_rpn:
                proposal_cfg = self.detector.train_cfg.get(
                    'rpn_proposal', self.detector.test_cfg.rpn)
                rpn_losses, proposal_list = \
                    self.detector.rpn_head.forward_train(
                        x,
                        img_metas,
                        gt_bboxes,
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            roi_losses = self.detector.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, **kwargs)
            losses.update(roi_losses)

        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            bbox_losses = self.detector.bbox_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        return losses

    # def simple_test(self,
    #                 img,
    #                 img_metas,
    #                 ref_img=None,
    #                 ref_img_metas=None,
    #                 proposals=None,
    #                 rescale=False):
    #     """Test without augmentation.
    #
    #     Args:
    #         img (list[Tensor]): of shape (1, C, H, W) encoding input image.
    #             Typically these should be mean centered and std scaled.
    #
    #         img_metas (list[dict]): list of image information dict where each
    #             dict has: 'img_shape', 'scale_factor', 'flip', and may also
    #             contain 'filename', 'ori_shape', 'pad_shape', and
    #             'img_norm_cfg'. For details on the values of these keys see
    #             `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
    #
    #         ref_img (list[Tensor] | None): The list only contains one Tensor
    #             of shape (1, N, C, H, W) encoding input reference images.
    #             Typically these should be mean centered and std scaled. N
    #             denotes the number for reference images. There may be no
    #             reference images in some cases.
    #
    #         ref_img_metas (list[list[list[dict]]] | None): The first and
    #             second list only has one element. The third list contains
    #             image information dict where each dict has: 'img_shape',
    #             'scale_factor', 'flip', and may also contain 'filename',
    #             'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
    #             the values of these keys see
    #             `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
    #             may be no reference images in some cases.
    #
    #         proposals (None | Tensor): Override rpn proposals with custom
    #             proposals. Use when `with_rpn` is False. Defaults to None.
    #
    #         rescale (bool): If False, then returned bboxes and masks will fit
    #             the scale of img, otherwise, returned bboxes and masks
    #             will fit the scale of original image shape. Defaults to False.
    #
    #     Returns:
    #         dict[str : list(ndarray)]: The detection results.
    #     """
    #     if isinstance(ref_img, list):
    #         ref_img = ref_img[0]
    #     all_imgs = torch.cat((ref_img[0], img), dim=0)
    #     # noise, clean = general_clean_noise_pairs(all_imgs, constant=[0.5, 0.5])
    #     # all_x = self.detector.extract_feat(noise)
    #     all_x = self.detector.extract_feat(all_imgs)
    #
    #     x = []
    #     for i in range(len(all_x)):  # index 0 is the noise features
    #         agg_x = self.aggregator(all_x[i][[-1]], all_x[i][:-1])
    #         x.append(agg_x)
    #
    #     # Two stage detector
    #     if hasattr(self.detector, 'roi_head'):
    #         if proposals is None:
    #             proposal_list = self.detector.rpn_head.simple_test_rpn(
    #                 x, img_metas)
    #         else:
    #             proposal_list = proposals
    #
    #         outs = self.detector.roi_head.simple_test(
    #             x, proposal_list, img_metas, rescale=rescale)
    #
    #     # Single stage detector
    #     elif hasattr(self.detector, 'bbox_head'):
    #         outs = self.bbox_head(x)
    #         bbox_list = self.bbox_head.get_bboxes(
    #             *outs, img_metas, rescale=rescale)
    #         # skip post-processing when exporting to ONNX
    #         if torch.onnx.is_in_onnx_export():
    #             return bbox_list
    #
    #         outs = [
    #             bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #             for det_bboxes, det_labels in bbox_list
    #         ]
    #     else:
    #         raise TypeError('detector must has roi_head or bbox_head.')
    #
    #     results = dict()
    #     results['bbox_results'] = outs[0]
    #     if len(outs) == 2:
    #         results['segm_results'] = outs[1]
    #     return results

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', 1)
        frame_stride = img_metas[0].get('frame_stride', 1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                self.memo.img = ref_img[0]
                ref_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])
            x = self.detector.extract_feat(img)
        # test with fixed stride
        else:
            if frame_id == 0:
                self.memo = Dict()
                ref_noise, ref_clean = general_clean_noise_pairs(ref_img[0], constant=[0.5, 0.5])
                self.memo.img = ref_clean
                ref_x = self.detector.extract_feat(ref_noise)
                # _, ref_x = self.detector.extract_feat(ref_clean)
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
                ref_noise, ref_clean = general_clean_noise_pairs(ref_img[0], constant=[0.5, 0.5])
                x = []
                ref_imgs = torch.cat((self.memo.img, ref_noise), dim=0)[1:]
                ref_x = self.detector.extract_feat(ref_imgs)
                for i in range(len(ref_x)):
                    self.memo.feats[i] = ref_x[i]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img = ref_imgs
            else:
                assert ref_img is None
                ref_noise, ref_clean = general_clean_noise_pairs(img, constant=[0.5, 0.5])
                x = self.detector.extract_feat(ref_noise)
                # _, x = self.detector.extract_feat(clean)

        agg_x = []
        for i in range(len(x)):
            ref_feats = torch.cat((self.memo.feats[i][:num_left_ref_imgs],
                                   self.memo.feats[i][num_left_ref_imgs+1:]), dim=0)
            # agg_x_single = self.aggregator(x[i], self.memo.feats[i])
            agg_x_single = self.aggregator(x[i], ref_feats)
            agg_x.append(agg_x_single)
        return agg_x

    def simple_test(self,
                      img,
                      img_metas,
                      ref_img=None,
                      ref_img_metas=None,
                      proposals=None,
                      rescale=False
                      ):
        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]
        x = self.extract_feats(img, img_metas, ref_img, ref_img_metas)

        # Two stage detector
        if hasattr(self.detector, 'roi_head'):
            if proposals is None:
                proposal_list = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)
            else:
                proposal_list = proposals

            outs = self.detector.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        # Single stage detector
        elif hasattr(self.detector, 'bbox_head'):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_metas, rescale=rescale)
            # skip post-processing when exporting to ONNX
            if torch.onnx.is_in_onnx_export():
                return bbox_list

            outs = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError