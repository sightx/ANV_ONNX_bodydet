import torch
import onnxruntime as ort
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse




class BBox:
    def __init__(self, bbox_as_list):
        self.x1, self.y1, self.x2, self.y2, self.score, self.classid = bbox_as_list

    def to_list(self):
        return [self.x1, self.y1, self.x2, self.y2, self.score, self.classid]

    def __repr__(self):
        return f"{self.__str__()}"

    def __str__(self):
        return f"[{self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.score}, {self.classid}]"

class NmsAnv:
    def __init__(self, threshold=.25):
        self.threshold = threshold

    def run(self, bboxes):
        bboxes_as_list = []
        for i in range(0, bboxes.shape[0]):
            bboxes_as_list.append([bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3], bboxes[i, 4], 0])

        bboxes = sorted([BBox(b) for b in bboxes_as_list], key=lambda a: a.score, reverse=True)
        num_bbox = len(bboxes)
        mask_merged = [False] * num_bbox
        select_idx = 0
        output = []
        while True:
            while select_idx < num_bbox and mask_merged[select_idx]:
                select_idx += 1
            if select_idx == num_bbox:
                break

            mask_merged[select_idx] = True
            output.append(bboxes[select_idx].to_list())
            for i in range(select_idx, num_bbox):
                if mask_merged[i] or bboxes[i].classid != bboxes[select_idx].classid:
                    continue

                x1 = max(bboxes[select_idx].x1, bboxes[i].x1)
                y1 = max(bboxes[select_idx].y1, bboxes[i].y1)
                x2 = min(bboxes[select_idx].x2, bboxes[i].x2)
                y2 = min(bboxes[select_idx].y2, bboxes[i].y2)

                if x2 < x1 or y2 < y1:
                    continue

                inter_area = (x2 - x1) * (y2 - y1)
                chosen_area = (bboxes[select_idx].x2 - bboxes[select_idx].x1) * (
                        bboxes[select_idx].y2 - bboxes[select_idx].y1)
                process_area = (bboxes[i].x2 - bboxes[i].x1) * (bboxes[i].y2 - bboxes[i].y1)
                iou = inter_area / (chosen_area + process_area - inter_area)

                # print(f"iou is {iou}")

                if iou > self.threshold:
                    mask_merged[i] = True

        temp = []
        for i in range(0, len(output)):
            temp.append([output[i][0], output[i][1], output[i][2], output[i][3], output[i][4]])

        return temp

def pre_processing_one_patch(input):
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

def pre_processing_two_patch(img):
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

def post_processing(output, a_nScoreThreshold, width, height, offset=False):
    pLocationData = output[0]
    pConfig = output[1]

    def create_PriorCfg_COCO_512():
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

    m_PriorConfig = create_PriorCfg_COCO_512()

    def generatePriors(m_PriorConfig):
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

    m_vPriors = generatePriors(m_PriorConfig)

    boxes = []

    for j in range(0, m_PriorConfig["priors_size"]):
        nScore = pConfig[0, j, 1]

        if nScore > a_nScoreThreshold:
            nBb0 = m_vPriors[j*4 + 0] + (pLocationData[0, j, 0] * m_PriorConfig["variance"][0] * m_vPriors[j*4 + 2])
            nBb1 = m_vPriors[j*4 + 1] + (pLocationData[0, j, 1] * m_PriorConfig["variance"][0] * m_vPriors[j*4 + 3])
            nBb2 = m_vPriors[j*4 + 2] * math.exp(pLocationData[0, j, 2] * m_PriorConfig["variance"][1])
            nBb3 = m_vPriors[j*4 + 3] * math.exp(pLocationData[0, j, 3] * m_PriorConfig["variance"][1])
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

def plot_bb_on_frame(bbs, frame):
    if len(bbs) > 0:
        bb_num = bbs.shape[0]
    else:
        bb_num = 0

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(frame)

    for i in range(0, bb_num):
        X1 = bbs[i, 0].astype("int")
        Y1 = bbs[i, 1].astype("int")
        X2 = bbs[i, 2].astype("int")
        Y2 = bbs[i, 3].astype("int")

        w = X2 - X1
        h = Y2 - Y1

        # Create a Rectangle patch
        rect = patches.Rectangle((X1, Y1), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')  # np.uint8
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close("all")

    return data

def print_to_file(frame_num, f, bbs):
    if len(bbs) > 0:
        total_string = str(frame_num) + " "

        bb_num = bbs.shape[0]

        for i in range(0, bb_num):
            the_class = 0
            score = bbs[i, 4]
            X1 = bbs[i, 0].astype("int")
            Y1 = bbs[i, 1].astype("int")
            X2 = bbs[i, 2].astype("int")
            Y2 = bbs[i, 3].astype("int")
            w = X2 - X1
            h = Y2 - Y1

            total_string += str(the_class) + "," + str(score) + "," + str(X1 + int(w / 2)) + "," + str(Y1 + int(h / 2)) + "," + str(w) + "," + str(h)

            if i != bb_num - 1:
                total_string += " "

        f.write(total_string + "\n")

def create_detections(data_path, args, two_patches=True):
    ort_session = ort.InferenceSession("detector.onnx")
    nms = NmsAnv(args.NMSThreshold)
    cap = cv2.VideoCapture(data_path)
    file_name = os.path.basename(data_path)

    if args.create_bbs_video:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(os.path.join(args.bbs_video_dir, file_name[:-4] + "_with_bboxes.avi"), fourcc, 25.0, (640, 480))

    f = open(os.path.join(args.bboxs_dir, file_name[:-4] + ".txt"), "w")
    ret, img = cap.read()

    count = 1
    while ret and count:
        if two_patches:
            height, width = img.shape[0], img.shape[1]
            offset = width - height
            img_to_print = img

            first_patch, second_patch = pre_processing_two_patch(img)
            first_ort_inputs = {ort_session.get_inputs()[0].name: first_patch.detach().cpu().numpy()}
            first_ort_outs = ort_session.run(None, first_ort_inputs)
            first_boxes_array = post_processing(output=first_ort_outs, a_nScoreThreshold=args.scoreThreshold, width=height, height=height)

            second_ort_inputs = {ort_session.get_inputs()[0].name: second_patch.detach().cpu().numpy()}
            second_ort_outs = ort_session.run(None, second_ort_inputs)
            second_boxes_array = post_processing(output=second_ort_outs, a_nScoreThreshold=args.scoreThreshold, width=height, height=height, offset=offset)

            first_bb_out = nms.run(first_boxes_array)
            second_bb_out = nms.run(second_boxes_array)

            bb_out = np.array(first_bb_out + second_bb_out)
            bb_out = np.array(nms.run(bb_out))

        else:
            height, width = img.shape[0], img.shape[1]
            img_to_print = img
            patch = pre_processing_one_patch(img)
            ort_inputs = {ort_session.get_inputs()[0].name: patch.detach().cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            boxes_array = post_processing(output=ort_outs, a_nScoreThreshold=args.scoreThreshold, width=width, height=height)
            bb_out = np.array(nms.run(boxes_array))

        print_to_file(count, f, bb_out)
        print("frame count: ", count, "number of detections:", len(bb_out))
        plt.close("all")
        if args.create_bbs_video:
            data = plot_bb_on_frame(bb_out, img_to_print)
            video_writer.write(data)
        ret, img = cap.read()
        count += 1

    cap.release()
    if args.create_bbs_video:
        video_writer.release()
    cv2.destroyAllWindows()
    f.close()

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="dataset_prep")
    parser.add_argument(
        "--data_path", help="Path dir of videos or path to video itself",
        default='/home/nivpekar/projects/data/videos/soi_outdoor_new')
    parser.add_argument(
        "--create_bbs_video", help="create a video with bboxes", default=True)
    parser.add_argument(
        "--bbs_video_dir", help="where to create the a videos with bboxes",
        default="")
    parser.add_argument(
        "--bboxs_dir", help="where to create the files of the bboxs",
        default="")
    parser.add_argument(
        "--two_patches", help="two_patches or one patch resized to 512",
        default=True)
    parser.add_argument(
        "--NMSThreshold", help="",
        default=0.1)
    parser.add_argument(
        "--scoreThreshold", help="",
        default=0.1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.NMSThreshold = float(args.NMSThreshold)
    args.scoreThreshold = float(args.scoreThreshold)
    isDirectory = os.path.isdir(args.data_path)
    if isDirectory != True:
        print("working on vid:", args.data_path, "\n\n")
        create_detections(args.data_path, args, two_patches=args.two_patches)
    else:
        if not os.path.exists(os.path.join(args.bboxs_dir, "bboxs_dir")):
            os.mkdir(os.path.join(args.bboxs_dir, "bboxs_dir"))
        args.bboxs_dir = os.path.join(args.bboxs_dir, "bboxs_dir")

        if args.create_bbs_video:
            if not os.path.exists(os.path.join(args.bbs_video_dir, "bbs_video_dir")):
                os.mkdir(os.path.join(args.bbs_video_dir, "bbs_video_dir"))
            args.bbs_video_dir = os.path.join(args.bbs_video_dir, "bbs_video_dir")

        for r, d, f in os.walk(args.data_path):
            for file in f:
                print("working on vid:", os.path.join(r, file), "\n\n")
                create_detections(data_path=os.path.join(r, file), args=args, two_patches=args.two_patches)









