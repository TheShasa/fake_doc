import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pytesseract
from pytesseract import Output
from helper_class import Helper


class IsFakePhoto:
    magic_words = ['driving', 'licence', 'state', 'israel']

    magic_words_area_mean = {'driving': 1876.666666666666,
                             'licence': 1935.6666666666667,
                             'state': 604.6666666666666,
                             'israel': 744.6666666666666}
    magic_words_area_std = {'driving': 114.77611056119456,
                            'licence': 114.43872693377108,
                            'state': 36.7544403968947,
                            'israel': 49.30404536028346}

    mean_colors = np.array([[71.33333333, 35.04166667, 123.75],
                            [98.9, 45.7, 202.9],
                            [95.6, 34.8, 205.2],
                            [70.02083333, 34.10416667, 150.66666667]])

    std_colors = np.array([[36.46192839, 17.50828177, 11.42092378],
                           [8.96046874, 14.69727866, 35.78672938],
                           [11.5082579, 14.35827288, 38.48324311],
                           [42.77484151, 17.91093658, 21.90446428]])

    img_size = (800, 500)

    def __init__(self, img):
        self.img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        self.ocr = None
        self.debug_area_words = []
        self.array_of_dicts = []
        self.debug_color_words = []

    def prepare_ocr_dict(self):
        self.ocr = pytesseract.image_to_data(self.img, output_type=Output.DICT,  # lang='hebrew',
                                             config='--oem 3')
        n_boxes = len(self.ocr['level'])

        for i in range(n_boxes):
            if int(self.ocr['conf'][i]) <= 0:
                continue

            (word, x, y, w, h) = (
                self.ocr['text'][i], self.ocr['left'][i], self.ocr['top'][i], self.ocr['width'][i],
                self.ocr['height'][i])
            word = word.replace(' ', '')
            word = word.replace('.', '')
            word = word.lower()

            dict_ = {
                'word': word,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            }
            if len(word) <= 2:
                continue

            self.array_of_dicts.append(dict_)

    @staticmethod
    def word_area(img, box):
        '''
        returns color and area of a word in a box
        :param img:
        :param box:
        :return:
        '''

        crop1 = Helper.crop_photo_area_pixel(img, box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])

        gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return np.int32(np.mean(crop1[th == 0], axis=0)), np.sum(th == 0)

    def our_word(self, word, x, y):
        '''
        select our words x,y position or by magic word
        '''
        position_to_check_bold = [((305, 420), (100, 210)),
                                  ((305, 420), (265, 330))
                                  ]
        for magic_word in self.magic_words:
            if word.lower() in magic_word or magic_word in word.lower():
                return True
        for pos in position_to_check_bold:
            x1, x2 = pos[0]
            y1, y2 = pos[1]

            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def word_in_ocr(self, word):
        for dict_ in self.array_of_dicts:
            ocr_word = dict_['word']
            if word.lower() in ocr_word or ocr_word in word.lower():
                return True
        return False

    def has_magic_words(self, debug=False):
        '''check if it has magic words in it'''
        if self.ocr is None:
            self.prepare_ocr_dict()
        for word in self.magic_words:
            if not self.word_in_ocr(word):
                if debug:
                    print(f'word {word} not found!')
                return False
        return True

    def width_is_ok(self, debug=False):
        '''
        check if boldness of word is ok
        '''
        if self.ocr is None:
            self.prepare_ocr_dict()
        res = True
        for dict_ in self.array_of_dicts:
            word = dict_['word']
            for magic_word in self.magic_words:
                if word in magic_word or magic_word in word.lower():
                    _, area = self.word_area(self.img, dict_)
                    mean = self.magic_words_area_mean[magic_word]
                    std = self.magic_words_area_std[magic_word]
                    if area > mean + 3 * std:
                        if debug:
                            self.debug_area_words.append((dict_, area))

                            print(f'Word {word},{magic_word} has width of {area}, upper boundary is {mean + 3 * std}')
                        res = False
                    if area < mean - 3 * std:
                        if debug:
                            self.debug_area_words.append((dict_, area))
                            print(f'Word {word} has width of {area}, lower boundary is {mean - 3 * std}')
                        res = False

        return res

    def color_is_ok(self, debug=False):
        '''check if color of words is ok
        '''
        im = self.img
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        if self.ocr is None:
            self.prepare_ocr_dict()
        res = True
        for dict_ in self.array_of_dicts:

            word = dict_['word']
            x = dict_['x']
            y = dict_['y']
            contains_digit = any(map(str.isdigit, word))
            contains_letters = word.lower().islower()

            if (contains_digit and contains_letters) or (not contains_letters and not contains_digit):
                continue

            mean, area = self.word_area(im_hsv, dict_)
            color_class = -1

            if self.our_word(word, x, y):
                if word.lower() == 'driving' or word.lower() == 'licence':
                    color_class = 1
                elif word.lower() == 'state' or word.lower() == 'israel':
                    color_class = 2
                else:
                    color_class = 0

            else:
                if y >= 200:
                    color_class = 3
            if color_class >= 0:
                mean_ = self.mean_colors[color_class]
                std_ = self.std_colors[color_class]
                if np.any(mean > mean_ + 3 * std_):
                    if debug:
                        print(f'Strange color of word {word} at position x={x}, y={y}, '
                              f'color = {mean}, upper = {mean_ + 3 * std_}')
                        self.debug_color_words.append((dict_, mean))
                    res = False
                if np.any(mean < mean_ - 3 * std_):
                    if debug:
                        print(f'Strange color of word {word} at position x={x}, y={y}, '
                              f'color = {mean}, lower_bound = {mean_ - 3 * std_}')
                        self.debug_color_words.append((dict_, mean))
                    res = False
        return res

    def save_debuged_image(self, img_name='debuged'):

        plt.figure(figsize=(20, 40))
        fig, ax = plt.subplots(1, figsize=(40, 20))
        ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        for dict_, area in self.debug_area_words:
            x, y = dict_['x'], dict_['y']
            w, h = dict_['w'], dict_['h']
            plt.text(x, y, str(x) + ' ' + str(y) + ' ' + str(area),
                     horizontalalignment='left', color='b', verticalalignment='bottom', fontsize=20)
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        for dict_, mean in self.debug_color_words:
            x, y = dict_['x'], dict_['y']
            w, h = dict_['w'], dict_['h']
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            plt.text(x, y + h, str(x) + ' ' + str(y) + ' ' + str(mean),
                     horizontalalignment='left', color='b', verticalalignment='top', fontsize=20)

        plt.savefig(img_name + '.png', bbox_inches='tight', pad_inches=0)
