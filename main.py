import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pytesseract
from pytesseract import Output

from utils import *


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


def get_photo_corners2(img):
    img_cropped = crop_photo_area(img, 0, 0, 0.3, 0.8)
    boxes = draw_boxes(img_cropped)
    for line in boxes:
        if line['word'] == 'STATE':
            print(line['x'], line['y'])

            right_top_crop, right_top_x, right_top_y = crop_from_point(img_cropped, (line['x'], line['y']), 0.40, 0.05,
                                                                       0.30, 0.1)
            left_top_crop, left_top_x, left_top_y = crop_from_point(img_cropped, (line['x'], line['y']), -0.40, 0.05,
                                                                    0.30, 0.1)
            left_bot_crop, left_bot_x, left_bot_y = crop_from_point(img_cropped, (line['x'], line['y']), -0.40, 0.65,
                                                                    0.20, 0.3)

            crops = [left_bot_crop, right_top_crop, left_top_crop]
            points = []
            for crop in crops:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x, y = one_corner(th)
                points.append((x, y))

            left_bot_i = points[0]
            right_top_i = points[1]
            left_top_i = points[2]

            left_top_i = convert_xy(left_top_i, left_top_x, left_top_y)
            left_bot_i = convert_xy(left_bot_i, left_bot_x, left_bot_y)
            right_top_i = convert_xy(right_top_i, right_top_x, right_top_y)

            right_bot_i = find_4_in_paralegram(left_bot_i, right_top_i, left_top_i)
            points_ = [left_top_i, right_top_i, left_bot_i, right_bot_i]
            return perspect(points_, img)


def get_photo_corners(img):
    img2 = img.copy()
    img_cropped = crop_photo_area(img, 0, 0, 0.3, 0.8)
    line_image = img_cropped.copy()
    (x1, y1, x2, y2) = get_lowest_line(img_cropped)

    # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    boxes = draw_boxes(img_cropped)

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
            if distance_to_line((x1, y1), (x2, y2), (x, y)) < 3:
                if x > max_x:
                    max_x = x
                    right_bot_i = i
            # cv2.circle(line_image,(x,y),3,255,-1)

    if left_top_i is not None and right_bot_i is not None:
        left_top_x, left_top_y = left_top_i.ravel()
        right_bot_x, right_bot_y = right_bot_i.ravel()

        left_bot_x, left_bot_y = projection_on_line((x1, y1), (x2, y2), (left_top_x, left_top_y))
        left_bot_x = int(left_bot_x)
        left_bot_y = int(left_bot_y)

        crop1 = crop_photo_area_pixel(img_cropped, left_top_x - 10, left_top_y - 10, left_bot_x + 10, left_bot_y + 10)
        crop2 = crop_photo_area_pixel(img_cropped, right_bot_x - 10, left_top_y - 10, right_bot_x + 10,
                                      right_bot_y + 10)

    left_top_i, left_bot_i = good_corners(crop1, [0, 3])
    right_top_i, = good_corners(crop2, [1])

    left_top_i = convert_xy(left_top_i.ravel(), left_top_x - 10, left_top_y - 10)
    left_bot_i = convert_xy(left_bot_i.ravel(), left_top_x - 10, left_top_y - 10)
    right_top_i = convert_xy(right_top_i.ravel(), right_bot_x - 10, left_top_y - 10)
    right_bot_i = find_4_in_paralegram(left_bot_i, right_top_i, left_top_i)

    left_top_i = convert_xy(left_top_i, 0, 0)
    left_bot_i = convert_xy(left_bot_i, 0, 0)
    right_top_i = convert_xy(right_top_i, 0, 0)
    right_bot_i = convert_xy(right_bot_i, 0, 0)

    list_ = [left_top_i, right_top_i, left_bot_i, right_bot_i]
    return perspect(list_, img)


def strange_words(im, debug=False, img_name='debuged'):
    mean_colors = np.array([[117.66666667, 111.6, 103.46666667],
                            [175.33333333, 120.83333333, 88.16666667],
                            [187., 148.5, 124.33333333],
                            [149.89285714, 143.46428571, 136.28571429]])

    std_colors = np.array([[11.46976702, 9.039174, 9.89859698],
                           [11.22002179, 11.89420961, 19.63344652],
                           [11.71893055, 7.15891053, 13.28742095],
                           [15.27916248, 13.36167799, 13.89978711]])
    d = pytesseract.image_to_data(im, output_type=Output.DICT,  # lang='hebrew',
                                  config='--oem 3')
    n_boxes = len(d['level'])
    array_of_dicts = []
    if debug:
        plt.figure(figsize=(20, 40))
        fig, ax = plt.subplots(1, figsize=(40, 20))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for i in range(n_boxes):
        if int(d['conf'][i]) <= 0:
            continue

        (word, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        word1 = word.replace(' ', '')
        word1 = word1.replace('.', '')

        contains_digit = any(map(str.isdigit, word1))
        contains_letters = word1.lower().islower()

        dict_ = {
            'word': word,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }
        if len(word1) <= 2:
            array_of_dicts.append(dict_)
            continue
        if (contains_digit and contains_letters) or (not contains_letters and not contains_digit):
            array_of_dicts.append(dict_)
            continue

        mean, area = word_area(im, dict_)
        color_class = -1
        plot = True
        if our_word(word, x, y):
            if word.lower() == 'driving' or word.lower() == 'licence':
                color_class = 1
            elif word.lower() == 'state' or word.lower() == 'israel':
                color_class = 2
            else:
                color_class = 0
            if debug:
                rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='g', facecolor='none')
        else:
            if y >= 200:
                color_class = 3
                if debug:
                    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='b', facecolor='none')

            else:
                plot = False
                pass
                # if debug:
                #     rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

        if debug and plot:
            plt.text(x, y, str(x) + ' ' + str(y) + ' ' + str(mean),
                     # str(mean) + ' ' + str(area)+ ' ' + str(word),
                     horizontalalignment='center', color='b', verticalalignment='bottom', fontsize=20)
        if debug:
            ax.add_patch(rect)

        if color_class >= 0:
            if np.any(mean > mean_colors[color_class] + 3 * std_colors[color_class]) or np.any(
                    mean < mean_colors[color_class] - 3 * std_colors[color_class]):
                print(
                    f'Strange word {word} at position x={x}, y={y}, color = {mean}, mean = {mean_colors[color_class]}')
            array_of_dicts.append(dict_)
    if debug:
        plt.savefig(img_name + '.png', bbox_inches='tight', pad_inches=0)
    return array_of_dicts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('wrong arguments')
        exit(1)
    img_name = sys.argv[1]

    img = cv2.imread(img_name)

    if True:
        points, cropped, tr = get_photo_corners(img)
        points2, cropped2, tr2 = get_photo_corners2(img)
        print(points)
        print(points2)
        print(tr, tr2)
        if tr < tr2:
            strange_words(cropped, True)
        else:
            strange_words(cropped2, True)
    # except:
    #     print('not cropped')
    #     strange_words(img, True)
