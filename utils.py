import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def crop_photo_area_pixel(img, left, top, right, bottom):
    return img[top:bottom, left:right, :]


def projection_on_line(point1, point2, point):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    x = point[0]
    y = point[1]

    A = y2 - y1
    B = -(x2 - x1)
    C = -x1 * y2 + x1 * y1 + x2 * y1 - x1 * y1

    Cnorm = B * x - A * y

    matrix = np.array([[A, B], [-B, A]])
    matrix_x = np.array([[-C, B], [-Cnorm, A]])
    matrix_y = np.array([[A, -C], [-B, -Cnorm]])
    x = np.linalg.det(matrix_x) / np.linalg.det(matrix)
    y = np.linalg.det(matrix_y) / np.linalg.det(matrix)
    return x, y


def draw_boxes(im, plot_image=False):
    d = pytesseract.image_to_data(im, output_type=Output.DICT,  # lang='hebrew',
                                  config='--oem 3')
    n_boxes = len(d['level'])
    array_of_dicts = []
    for i in range(n_boxes):
        if int(d['conf'][i]) <= 0:
            continue

        (word, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])

        if word == '' or word == ' ':
            continue
        dict_ = {
            'word': word,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }

        array_of_dicts.append(dict_)
    return array_of_dicts


def angle(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    x_catet = x2 - x1
    y_catet = y2 - y1
    if x_catet == 0:
        return np.pi / 2
    return abs(np.arctan(y_catet / x_catet))


def get_lowest_line(img_cropped):
    (mu, sigma) = cv2.meanStdDev(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(img_cropped, mu - sigma, mu + sigma)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)

    max_y = 0
    max_line = None
    for line in lines:
        rho, theta = line[0][0], line[0][1]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if angle((x1, y1), (x2, y2)) < np.pi / 20:
            if y0 > max_y:
                max_y = y0
                max_line = line
    #             print((x1,y1),(x2,y2))
    #             print(angle((x1,y1),(x2,y2)))
    #             cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if max_line is not None:
        rho, theta = max_line[0][0], max_line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return (x1, y1, x2, y2)
    return (0, 0, 0, 0)


def distance_to_line(point1, point2, point):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    x = point[0]
    y = point[1]

    A = y2 - y1
    B = -(x2 - x1)
    C = -x1 * y2 + x1 * y1 + x2 * y1 - x1 * y1

    dist = abs((A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2))
    return dist


def crop_photo_area(img, left_perc, top_perc, right_perc, bot_perc):
    height = img.shape[0]
    width = img.shape[1]
    left = int(left_perc * width)
    right = int(right_perc * width)

    top = int(top_perc * height)
    bottom = int(bot_perc * height)

    return img[top:bottom, left:right, :]


def good_corners(img, corner_numbers=[]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = cv2.goodFeaturesToTrack(th, 155, 0.1, 10)
    corners = np.int0(corners)
    res = []
    for corner_num in corner_numbers:
        res.append(find_corner(th, corners, corner_num))
    return res


def find_corner(img, corners, corner_num):
    '''
        corner_num = 0 - left top
        corner_num = 1  - right top
        corner_num = 2 - right_bottom
        corner_num = 3 - left_bottom
    '''
    shape = img.shape

    min_ = 0
    min_i = None
    for i in corners:

        x, y = i.ravel()
        value = 0
        if corner_num == 0:
            value = x + y
        if corner_num == 1:
            value = y + shape[0] - x
        if corner_num == 2:
            value = shape[1] - y + shape[0] - x
        if corner_num == 3:
            value = x + shape[1] - y
        if value < min_ or min_ == 0:
            min_ = value
            min_i = i

    return min_i


def convert_xy(point, start_x, start_y):
    '''

     point - point we want to convert
     start_x - left_top_x of small img in big_img
     start_y - left_top_y of small img in big_img
    '''

    x = point[0]
    y = point[1]
    return (x + start_x, y + start_y)


def find_4_in_paralegram(first1, first2, second1):
    '''
        first 1 and first2 - two opposite points
        second 2 - points whcih we does not know the opposite
    '''
    f1_x, f1_y = first1
    f2_x, f2_y = first2
    s1_x, s1_y = second1

    center_x = (f1_x + f2_x) / 2
    center_y = (f1_y + f2_y) / 2

    s2_x = center_x + (center_x - s1_x)
    s2_y = center_y + (center_y - s1_y)

    return int(s2_x), int(s2_y)


def our_word(word, x, y):
    words_to_check = ['driving', 'licence', 'state', 'israel']

    position_to_check_bold = [((305, 420), (100, 210)),
                              ((305, 420), (265, 330))
                              ]
    for magic_word in words_to_check:
        if word.lower() in magic_word or magic_word in word.lower():
            return True
    for pos in position_to_check_bold:
        x1, x2 = pos[0]
        y1, y2 = pos[1]

        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def word_area(img, box):
    crop1 = crop_photo_area_pixel(img, box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])

    gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return np.int32(np.mean(crop1[th == 0], axis=0)), np.sum(th == 0)


def crop_from_point(img, point, perc_x, perc_y, size_x, size_y):
    x, y = point
    height = img.shape[0]
    width = img.shape[1]

    left = x + int(perc_x * width)
    right = left + int(size_x * width)

    top = y + int(perc_y * height)
    bottom = top + int(size_y * height)

    left = max(left, 0)
    right = min(right, width)
    top = max(top, 0)
    bottom = min(height, bottom)

    return img[top:bottom, left:right],left,top


def one_corner(th):
    height = th.shape[0]
    width = th.shape[1]
    corners = cv2.goodFeaturesToTrack(th, 155, 0.1, 10)
    if corners is not None:
        corners = np.int0(corners)
        x, y = find_closest_to_center(th, corners)

        return x, y
    return int(width / 2), int(height / 2)


def find_closest_to_center(img, corners):
    height = img.shape[0]
    width = img.shape[1]
    center_x, center_y = width / 2, height / 2

    min_d = -1;
    min_x = 0
    min_y = 0
    for corner in corners:
        x, y = corner.ravel()
        dist = np.linalg.norm((center_x - x, center_y - y))
        if dist < min_d or min_d == -1:
            min_d = dist
            min_x = x
            min_y = y
    return min_x, min_y
