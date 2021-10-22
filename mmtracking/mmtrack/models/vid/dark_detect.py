import torch
import torch.nn.functional as F
from addict import Dict

from mmdet.core import bbox2result
# from mmtrack.core.utils import general_clean_noise_pairs, plot_noise_clean_pairs
from ..builder import MODELS, build_detector, build_aggregator, build_cleaner
from .base import BaseVideoDetector


@MODELS.register_module()
class DarkDetect(BaseVideoDetector):
    """
    Detect for low light videos.
    """

    def __init__(self,
                 detector,
                 aggregator,
                 cleaner,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DarkDetect, self).__init__()
        self.detector = build_detector(detector)
        self.aggregator = build_aggregator(aggregator)
        self.cleaner = build_cleaner(cleaner)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), "'pretrain' must be a dict."
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_cleaner and pretrain.get('cleaner', False):
            self.init_module('cleaner', pretrain['cleaner'])

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
                      **kwargs
                      ):

        assert len(img) == 1, "video detector only support 1 batch size per gpu for now."

        index = ref_img.shape[1] // 2
        all_imgs = torch.cat((ref_img[0][:index], img, ref_img[0][index:]), dim=0)

        # noise, clean = img[0, :3, ...], img[0, 3:, ...]
        # noisep = noise.permute(1, 2, 0).cpu()
        # cleanp = clean.permute(1, 2, 0).cpu()

        # import matplotlib.pylab as plt
        # import numpy as np
        # noise_ = np.asarray(noisep)
        # clean_ = np.asarray(cleanp)
        # # noise_ = noise_.astype(np.float32)
        # # noise_ = np.clip(noise_, 0, 255.0)
        # # noise_ = noise_.astype(np.uint8)
        # fig = plt.figure()
        # no = plt.subplot2grid((1, 2), (0, 0))
        # no_ = no.imshow(noise_[:, :, ::-1])
        # cl = plt.subplot2grid((1, 2), (0, 1))
        # cl_ = cl.imshow(clean_[:, :, ::-1])
        # # fig.colorbar(cl_, ax=[no , cl])
        # plt.show()


        # all_imgs = torch.cat((ref_img[0], img), dim=0)
        noise, clean = general_clean_noise_pairs(all_imgs, constant=[0.5, 0.5])

        x_noise, all_x = self.detector.extract_feat(noise)
        x_clean = self.cleaner(clean)

        # plot x_noise and x_clean
        with torch.no_grad():
            for i in range(len(x_noise)):
                plot_noise_clean_pairs(x_clean[i][:, 0, ...], x_noise[i][:, 0, ...],
                                       save_dir=f'./plot/clean_noise_features_s{i}_1.jpg')
                plot_noise_clean_pairs(torch.sum(x_clean[i], dim=1, keepdim=False),
                                       torch.sum(x_noise[i], dim=1, keepdim=False),
                                       save_dir=f'./plot/clean_noise_features_s{i}_s.jpg')

        x = []
        for i in range(len(all_x)):  # index 0 is the noise features
            # agg_ref_x = torch.cat((all_x[i][:index], all_x[i][index+1:]), dim=0)
            agg_x = self.aggregator(all_x[i][[index]], all_x[i])
            # agg_x = self.aggregator(all_x[i][[-1]], all_x[i][:-1])
            x.append(agg_x)

        losses = dict()
        # clean and noise mse loss
        if self.with_cleaner:
            mse_losses = dict()
            # assert len(x_noise) == 4
            # rat = [0.1, 0.2, 0.3, 0.4]
            for i in range(len(x_noise)):
                # mse_losses[f'loss_mse_{i}'] = F.mse_loss(x_noise[i], x_clean[i]) * rat[i]
                mse_losses[f'loss_mse_{i}'] = F.mse_loss(x_noise[i], x_clean[i])
            losses.update(mse_losses)

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
                self.memo.img = ref_noise
                _, ref_x = self.detector.extract_feat(ref_noise)
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
                _, ref_x = self.detector.extract_feat(ref_imgs)
                for i in range(len(ref_x)):
                    self.memo.feats[i] = ref_x[i]
                    x.append(self.memo.feats[i][[num_left_ref_imgs]])
                self.memo.img = ref_imgs
            else:
                assert ref_img is None
                ref_noise, ref_clean = general_clean_noise_pairs(img, constant=[0.5, 0.5])
                _, x = self.detector.extract_feat(ref_noise)

        agg_x = []
        for i in range(len(x)):
            # ref_feats = torch.cat((self.memo.feats[i][:num_left_ref_imgs],
            #                        self.memo.feats[i][num_left_ref_imgs+1:]), dim=0)
            agg_x_single = self.aggregator(x[i], self.memo.feats[i])
            # agg_x_single = self.aggregator(x[i], ref_feats)
            agg_x.append(agg_x_single)
        return agg_x

    # def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
    #     frame_id = img_metas[0].get('frame_id', -1)
    #     assert frame_id >= 0
    #     num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', 1)
    #     frame_stride = img_metas[0].get('frame_stride', 1)
    #
    #     men_feats_len = 32
    #     x_index = num_left_ref_imgs
    #
    #     # test with adaptive stride
    #     if frame_id == 0:
    #         self.memo = Dict()
    #         ref_noise, ref_clean = general_clean_noise_pairs(ref_img[0], constant=[0.5, 0.5])
    #         self.memo.img = ref_noise
    #         _, ref_x = self.detector.extract_feat(ref_noise)
    #         # _, ref_x = self.detector.extract_feat(ref_clean)
    #         # 'tuple' object (e.g. the output of FPN) does not support
    #         # item assignment
    #         self.memo.feats = []
    #         # the features of img is same as ref_x[i][[num_left_ref_imgs]]
    #         x = []
    #         for i in range(len(ref_x)):
    #             self.memo.feats.append(ref_x[i])
    #             x.append(ref_x[i][[num_left_ref_imgs]])
    #             x_index = num_left_ref_imgs
    #     else:
    #         assert ref_img is not None
    #         ref_noise, ref_clean = general_clean_noise_pairs(ref_img[0], constant=[0.5, 0.5])
    #         x = []
    #         ref_imgs = torch.cat((self.memo.img, ref_noise), dim=0)[1:]
    #         _, ref_x = self.detector.extract_feat(ref_imgs)
    #         for i in range(len(ref_x)):
    #             if self.memo.feats[i].shape[0] < men_feats_len:
    #                 self.memo.feats[i] = torch.cat(
    #                     [self.memo.feats[i][:-(num_left_ref_imgs * 2)], ref_x[i]])
    #                 x_index = x_index + 1
    #                 x.append(self.memo.feats[i][[x_index]])
    #             else:
    #                 self.memo.feats[i] = torch.cat(
    #                     [self.memo.feats[i][:-(num_left_ref_imgs * 2)], ref_x[i]])[1:]
    #                 x_index = men_feats_len - num_left_ref_imgs
    #                 x.append(self.memo.feats[i][[x_index]])
    #         self.memo.img = ref_imgs
    #
    #     agg_x = []
    #     for i in range(len(x)):
    #         ref_feats = torch.cat((self.memo.feats[i][:x_index],
    #                                self.memo.feats[i][x_index+1:]), dim=0)
    #         # agg_x_single = self.aggregator(x[i], self.memo.feats[i])
    #         agg_x_single = self.aggregator(x[i], ref_feats)
    #         agg_x.append(agg_x_single)
    #     return agg_x

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

    def aug_test(self):
        """Test function with test time augmentation."""
        raise NotImplementedError