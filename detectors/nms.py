

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