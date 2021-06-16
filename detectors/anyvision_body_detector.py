from .abstract_detector import AbstractDetector
from .nms import NmsAnv
import torch
import onnxruntime as ort
import numpy as np
import math
import cv2


'''
ANV detector

example:

detector = ANVBodyDetector(one_patch_or_two_patch="two_patch",
                NMS_threshold=0.1,
                score_threshold=0.1,
                model_path='blablabla/ANV_ONNX_bodydet/detector_fp16.onnx',
                input_dtype="fp16",
                resize_dsize=(910, 512),
                detections_path="blablabla/some_file.txt"
                )


'''
class ANVBodyDetector(AbstractDetector):

    def __init__(self, one_patch_or_two_patch="two_patch", NMS_threshold=0.1, score_threshold=0.1, model_path=None, input_dtype=None, resize_dsize=None, detections_path=None):
        if one_patch_or_two_patch != "two_patch" and one_patch_or_two_patch != "one_patch":
            raise Exception('one_patch_or_two_patch most be "one_patch" or "two_patch" string, instead got:', one_patch_or_two_patch)
        self.one_patch_or_two_patch = one_patch_or_two_patch
        self.score_threshold = score_threshold
        if input_dtype != "fp16" and input_dtype != "float32":
            raise Exception('input_dtype most be "fp16" or "float32" string, instead got:', input_dtype)
        self.input_dtype = np.half if input_dtype == "fp16" else np.float32
        self.model = ort.InferenceSession(model_path)
        self.m_PriorConfig = self._create_PriorCfg_COCO_512()
        self.m_vPriors = self._generatePriors(self.m_PriorConfig)
        self.nms = NmsAnv(NMS_threshold)
        self.resize_dsize = resize_dsize
        self.detections_path = detections_path

    def _create_PriorCfg_COCO_512(self):
        PriorCfg_COCO_512 = {}
        PriorCfg_COCO_512["elements_len"] = 7
        PriorCfg_COCO_512["feature_maps"] = [64, 32, 16, 8, 4, 2, 1]
        PriorCfg_COCO_512["min_dim"] = 512
        PriorCfg_COCO_512["steps"] = [8, 16, 32, 64, 128, 256, 512]
        PriorCfg_COCO_512["min_sizes"] = [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8]
        PriorCfg_COCO_512["max_sizes"] = [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72]
        PriorCfg_COCO_512["aspect_ratios"] = []
        PriorCfg_COCO_512["variance"] = [0.1, 0.2]
        PriorCfg_COCO_512["clip"] = True
        PriorCfg_COCO_512["clip_max"] = 1.0
        PriorCfg_COCO_512["clip_min"] = 0
        PriorCfg_COCO_512["priors_size"] = 32756
        for i in range(0, 5):
            PriorCfg_COCO_512["aspect_ratios"].append([2, 3])
        PriorCfg_COCO_512["aspect_ratios"].append([2])
        PriorCfg_COCO_512["aspect_ratios"].append([2])
        return PriorCfg_COCO_512

    def _generatePriors(self, m_PriorConfig):
        m_vPriors = []
        for k in range(0, m_PriorConfig["elements_len"]):
            f = m_PriorConfig["feature_maps"][k]
            for i in range(0, f):
                for j in range(0, f):
                    f_k = float(m_PriorConfig["min_dim"]) / m_PriorConfig["steps"][k]
                    cx = (j + float(0.5)) / f_k
                    cy = (i + float(0.5)) / f_k
                    s_k = m_PriorConfig["min_sizes"][k] / m_PriorConfig["min_dim"]
                    m_vPriors.append(cx)
                    m_vPriors.append(cy)
                    m_vPriors.append(s_k)
                    m_vPriors.append(s_k)

                    s_k_prime = math.sqrt(s_k * m_PriorConfig["max_sizes"][k] / m_PriorConfig["min_dim"])
                    m_vPriors.append(cx)
                    m_vPriors.append(cy)
                    m_vPriors.append(s_k_prime)
                    m_vPriors.append(s_k_prime)

                    for ar in m_PriorConfig["aspect_ratios"][k]:
                        sqt_ar = math.sqrt(ar)

                        m_vPriors.append(cx)
                        m_vPriors.append(cy)
                        m_vPriors.append(s_k * sqt_ar)
                        m_vPriors.append(s_k / sqt_ar)
                        m_vPriors.append(cx)
                        m_vPriors.append(cy)
                        m_vPriors.append(s_k / sqt_ar)
                        m_vPriors.append(s_k * sqt_ar)
        for i in range(0, len(m_vPriors)):
            m_vPriors[i] = max(m_PriorConfig["clip_min"], min(m_vPriors[i], m_PriorConfig["clip_max"]))

        return m_vPriors

    def _post_processing(self, output, a_nScoreThreshold, width, height, offset=False):
        pLocationData = output[0]
        pConfig = output[1]
        boxes = []

        for j in range(0, self.m_PriorConfig["priors_size"]):
            nScore = pConfig[0, j, 1]

            if nScore > a_nScoreThreshold:
                nBb0 = self.m_vPriors[j * 4 + 0] + (
                            pLocationData[0, j, 0] * self.m_PriorConfig["variance"][0] * self.m_vPriors[j * 4 + 2])
                nBb1 = self.m_vPriors[j * 4 + 1] + (
                            pLocationData[0, j, 1] * self.m_PriorConfig["variance"][0] * self.m_vPriors[j * 4 + 3])
                nBb2 = self.m_vPriors[j * 4 + 2] * math.exp(pLocationData[0, j, 2] * self.m_PriorConfig["variance"][1])
                nBb3 = self.m_vPriors[j * 4 + 3] * math.exp(pLocationData[0, j, 3] * self.m_PriorConfig["variance"][1])
                nBb0 -= nBb2 / 2
                nBb1 -= nBb3 / 2
                nBb2 += nBb0
                nBb3 += nBb1
                nBb0 *= float(width)
                nBb1 *= float(height)
                nBb2 *= float(width)
                nBb3 *= float(height)
                bbox = {}
                bbox["X1"] = nBb0
                bbox["Y1"] = nBb1
                bbox["X2"] = nBb2
                bbox["Y2"] = nBb3
                bbox["Score"] = nScore

                boxes.append(bbox)

        boxes_array = np.empty((len(boxes), 5))
        for i in range(0, len(boxes)):
            boxes_array[i, 0] = boxes[i]["X1"]
            boxes_array[i, 1] = boxes[i]["Y1"]
            boxes_array[i, 2] = boxes[i]["X2"]
            boxes_array[i, 3] = boxes[i]["Y2"]
            if offset:
                boxes_array[i, 0] += offset
                boxes_array[i, 2] += offset
            boxes_array[i, 4] = boxes[i]["Score"]

        return boxes_array

    def _pre_processing_two_patch(self, img):
        resize_width = 910
        resized_img = cv2.resize(img, dsize=(resize_width, 512), interpolation=cv2.INTER_LINEAR)
        resized_img = np.transpose(resized_img, axes=[2, 0, 1])

        def temp_func(img):
            batch = torch.from_numpy(img)
            batch = torch.unsqueeze(batch, dim=0)
            input = batch.type(torch.float32)
            mean = torch.tensor([104, 117, 123])
            for i in range(0, 3):
                input[0, i] = input[0, i] - mean[i]
            return input

        resized_img = temp_func(resized_img)
        first_patch = resized_img[:, :, :, 0:512]
        second_patch = resized_img[:, :, :, (resize_width - 512):]

        return (first_patch, second_patch)

    def _pre_processing_one_patch(self, input):
        batch = cv2.resize(input, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        batch = torch.from_numpy(np.transpose(batch, axes=[2, 0, 1]))
        batch = torch.unsqueeze(batch, dim=0)

        if batch.shape != (1, 3, 512, 512):
            raise ValueError('input not in shape (1, 3, 512, 512)')

        input = batch.type(torch.float32)
        mean = torch.tensor([104, 117, 123])
        for i in range(0, 3):
            input[0, i] = input[0, i] - mean[i]
        return input

    '''
        input
              ----------
              image : ndarray

              output
              ----------
              output: ndarray  
              represent the bboxes. in the format of:
              number of bboxes = ouput.shape[0]

              for row i (bbox i):
                  object class number = bbs[i, 0]
                  score = = bbs[i, 1]
                  x center = bbs[i, 2]
                  y center = bbs[i, 3]
                  w = bbs[i, 4]
                  h = bbs[i, 5]

        '''
    def detect(self, img):
        if self.resize_dsize != None:
            img = cv2.resize(img, dsize=self.resize_dsize, interpolation=cv2.INTER_LINEAR)
        height, width = img.shape[0], img.shape[1]

        if self.one_patch_or_two_patch == "two_patch":
            offset = width - height
            first_patch, second_patch = self._pre_processing_two_patch(img)
            first_ort_inputs = {
                self.model.get_inputs()[0].name: first_patch.detach().cpu().numpy().astype(self.input_dtype)}
            first_ort_outs = self.model.run(None, first_ort_inputs)
            first_boxes_array = self._post_processing(output=first_ort_outs, a_nScoreThreshold=self.score_threshold,
                                                width=height, height=height)

            second_ort_inputs = {
                self.model.get_inputs()[0].name: second_patch.detach().cpu().numpy().astype(self.input_dtype)}
            second_ort_outs = self.model.run(None, second_ort_inputs)
            second_boxes_array = self._post_processing(output=second_ort_outs, a_nScoreThreshold=self.score_threshold,
                                                 width=height, height=height, offset=offset)

            first_bb_out = self.nms.run(first_boxes_array)
            second_bb_out = self.nms.run(second_boxes_array)

            bb_out = np.array(first_bb_out + second_bb_out)
            bb_out = np.array(self.nms.run(bb_out))

        else:
            patch = self._pre_processing_one_patch(img)
            ort_inputs = {self.model.get_inputs()[0].name: patch.detach().cpu().numpy().astype(self.input_dtype)}
            ort_outs = self.model.run(None, ort_inputs)
            boxes_array = self._post_processing(output=ort_outs, a_nScoreThreshold=self.score_threshold, width=width,
                                          height=height)
            bb_out = np.array(self.nms.run(boxes_array))

        # clamp detections
        if len(bb_out) > 0:
            bb_out[:, 0] = np.clip(bb_out[:, 0], 0, width)
            bb_out[:, 1] = np.clip(bb_out[:, 1], 0, height)
            bb_out[:, 2] = np.clip(bb_out[:, 2], 0, width)
            bb_out[:, 3] = np.clip(bb_out[:, 3], 0, height)

        if len(bb_out) > 0:
            reformatted_detections = np.zeros((len(bb_out), 6))
            x1 = bb_out[:, 0]
            y1 = bb_out[:, 1]
            x2 = bb_out[:, 2]
            y2 = bb_out[:, 3]
            w = x2 - x1
            h = y2 - y1
            reformatted_detections[:, 0] = 0                     # object class number
            reformatted_detections[:, 1] = bb_out[:, 4]          # score
            reformatted_detections[:, 2] = x1 + w / 2            # x center
            reformatted_detections[:, 3] = y1 + h / 2            # y center
            reformatted_detections[:, 4] = w                     # w
            reformatted_detections[:, 5] = h                     # h
        else:
            reformatted_detections = np.array([])

        return reformatted_detections
