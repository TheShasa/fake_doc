import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pytesseract
from pytesseract import Output
from helper_class import Helper


class IsFakePhoto:
    magic_words = ['driving', 'licence', 'state', 'israel']

    magic_words_area_mean = {'driving': 1849.2,
                             'licence': 1928.2,
                             'state': 597.8,
                             'israel': 728.8}
    magic_words_area_std = {'driving': 95.52884381169909,
                            'licence': 96.8574209856942,
                            'state': 52.51818732591596,
                            'israel': 47.9432998447124}

    mean_colors = np.array([[103.625, 102.58333333, 99.16666667],
                            [158.2, 106.3, 84.],
                            [169.8, 137.1, 120.8],
                            [127.33333333, 126.02083333, 124.97916667]])

    std_colors = np.array([[30.74330889, 23.57597058, 17.97838208],
                           [31.13454673, 21.72579112, 20.174241],
                           [33.54042337, 21.3703065, 13.81158934],
                           [46.63481056, 38.45695081, 31.17256378]])

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

            mean, area = self.word_area(im, dict_)
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
            plt.text(x, y, str(x) + ' ' + str(y) + ' ' + str(area),
                     horizontalalignment='left', color='b', verticalalignment='bottom', fontsize=20)
        for dict_, mean in self.debug_color_words:
            x, y = dict_['x'], dict_['y']
            w, h = dict_['w'], dict_['h']
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            plt.text(x, y + h, str(x) + ' ' + str(y) + ' ' + str(mean),
                     horizontalalignment='left', color='b', verticalalignment='top', fontsize=20)

        plt.savefig(img_name + '.png', bbox_inches='tight', pad_inches=0)
