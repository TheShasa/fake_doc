import cv2
import numpy as np

from helper_class import Helper


class PhotoCrop:
    def __init__(self, img):
        self.img = img
        self.points, self.cropped, self.tr = self.get_photo_corners()
        self.points2, self.cropped2, self.tr2 = self.get_photo_corners2()

        self.correct_crop = None

    def get_correct_crop(self):
        self.points, self.cropped, self.tr = self.get_photo_corners()
        self.points2, self.cropped2, self.tr2 = self.get_photo_corners2()

        self.correct_crop = self.img
        if 0 < self.tr < self.tr2:
            self.correct_crop = self.cropped
        elif self.tr2 > 0:
            self.correct_crop = self.cropped2
        return self.correct_crop

    @staticmethod
    def perspect(points_, img):
        pts1 = [0, 0, 0, 0]
        pts2 = [0, 0, 0, 0]
        ideals = [(0.052083333333333336, 0.13626373626373625),
                  (0.4305555555555556, 0.13626373626373625),
                  (0.052083333333333336, 0.4593406593406593),
                  (0.4305555555555556, 0.4593406593406593)]
        for i in range(4):
            pts1[i] = points_[i]
            pts2[i] = ideals[i][0] * img.shape[0], ideals[i][1] * img.shape[1]

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

        for i in range(4):
            x, y = ideals[i][0] * img.shape[0], ideals[i][1] * img.shape[1]
            cv2.circle(result, (int(x), int(y)), 3, 125, -1)
        return points_, result, np.trace(np.array(matrix))

    def get_photo_corners(self):
        img = self.img
        img_cropped = Helper.crop_photo_area(img, 0, 0, 0.3, 0.8)
        (x1, y1, x2, y2) = Helper.get_lowest_line(img_cropped)
        boxes = Helper.draw_boxes(img_cropped)

        max_y = np.max([i['y'] + i['h'] for i in boxes])

        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 155, 0.1, 10)
        corners = np.int0(corners)

        left_top_i = None
        min_y = 0

        max_x = 0
        right_bot_i = None
        for i in corners:
            x, y = i.ravel()
            if y > max_y:
                if y < min_y or min_y == 0:
                    min_y = y
                    left_top_i = i
                if Helper.distance_to_line((x1, y1), (x2, y2), (x, y)) < 3:
                    if x > max_x:
                        max_x = x
                        right_bot_i = i
                # cv2.circle(line_image,(x,y),3,255,-1)

        if left_top_i is not None and right_bot_i is not None:
            left_top_x, left_top_y = left_top_i.ravel()
            right_bot_x, right_bot_y = right_bot_i.ravel()

            left_bot_x, left_bot_y = Helper.projection_on_line((x1, y1), (x2, y2), (left_top_x, left_top_y))
            left_bot_x = int(left_bot_x)
            left_bot_y = int(left_bot_y)

            crop1 = Helper.crop_photo_area_pixel(img_cropped, left_top_x - 10, left_top_y - 10, left_bot_x + 10,
                                                 left_bot_y + 10)
            crop2 = Helper.crop_photo_area_pixel(img_cropped, right_bot_x - 10, left_top_y - 10, right_bot_x + 10,
                                                 right_bot_y + 10)
        else:
            return [], img, 0
        left_top_i, left_bot_i = Helper.good_corners(crop1, [0, 3])
        right_top_i, = Helper.good_corners(crop2, [1])

        left_top_i = Helper.convert_xy(left_top_i.ravel(), left_top_x - 10, left_top_y - 10)
        left_bot_i = Helper.convert_xy(left_bot_i.ravel(), left_top_x - 10, left_top_y - 10)
        right_top_i = Helper.convert_xy(right_top_i.ravel(), right_bot_x - 10, left_top_y - 10)
        right_bot_i = Helper.find_4_in_paralegram(left_bot_i, right_top_i, left_top_i)

        left_top_i = Helper.convert_xy(left_top_i, 0, 0)
        left_bot_i = Helper.convert_xy(left_bot_i, 0, 0)
        right_top_i = Helper.convert_xy(right_top_i, 0, 0)
        right_bot_i = Helper.convert_xy(right_bot_i, 0, 0)

        list_ = [left_top_i, right_top_i, left_bot_i, right_bot_i]
        return self.perspect(list_, img)

    def get_photo_corners2(self):
        img = self.img
        img_cropped = Helper.crop_photo_area(img, 0, 0, 0.3, 0.8)
        boxes = Helper.draw_boxes(img_cropped)
        for line in boxes:
            if line['word'] == 'STATE':

                right_top_crop, right_top_x, right_top_y = Helper.crop_from_point(img_cropped, (line['x'], line['y']),
                                                                                  0.40,
                                                                                  0.05,
                                                                                  0.30, 0.1)
                left_top_crop, left_top_x, left_top_y = Helper.crop_from_point(img_cropped, (line['x'], line['y']),
                                                                               -0.40,
                                                                               0.05,
                                                                               0.30, 0.1)
                left_bot_crop, left_bot_x, left_bot_y = Helper.crop_from_point(img_cropped, (line['x'], line['y']),
                                                                               -0.40,
                                                                               0.65,
                                                                               0.20, 0.3)

                crops = [left_bot_crop, right_top_crop, left_top_crop]
                points = []
                for crop in crops:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    x, y = Helper.one_corner(th, Helper.find_closest_to_center)
                    points.append((x, y))

                left_bot_i = points[0]
                right_top_i = points[1]
                left_top_i = points[2]

                left_top_i = Helper.convert_xy(left_top_i, left_top_x, left_top_y)
                left_bot_i = Helper.convert_xy(left_bot_i, left_bot_x, left_bot_y)
                right_top_i = Helper.convert_xy(right_top_i, right_top_x, right_top_y)

                right_bot_i = Helper.find_4_in_paralegram(left_bot_i, right_top_i, left_top_i)
                points_ = [left_top_i, right_top_i, left_bot_i, right_bot_i]
                return self.perspect(points_, img)
        return [], img, 0
