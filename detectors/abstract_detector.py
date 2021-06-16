from abc import ABC, abstractmethod
import numpy as np
import os.path


class AbstractDetector(ABC):


    '''
    detect is an abstract method, every detector should inherent from AbstractDetector and implement this method.

    input
          ----------
          image : ndarray

          output
          ----------
          ouput: ndarray
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
    @abstractmethod
    def detect(self, image):
        pass

    '''
                  input
                  ----------

                  frame_number : int
                    frame number in video

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
    def upload_frame_detections_from_file(self, frame_number):
        if not os.path.isfile(self.detections_path):
            raise Exception('detections file path does not exist:', self.detections_path)

        external_det_f = open(self.detections_path, 'r')
        for line in external_det_f:
            detections = line.split(' ')
            frame_id = int(detections[0])
            if frame_id == frame_number:
                detection_list = []
                for d_string in detections[1:]:
                    d = [float(v) for v in d_string.split(',')]
                    object_class = float(d[0])
                    score = float(d[1])
                    center_x = float(d[2])
                    center_y = float(d[3])
                    w = float(d[4])
                    h = float(d[5])
                    detection_list.append([object_class, score, center_x, center_y, w, h])
                return np.array(detection_list)

            else:
                if frame_id > frame_number:
                    return np.array([])
