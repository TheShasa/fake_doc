import cv2
from fake_class import IsFakePhoto
from crop_class import PhotoCrop

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('wrong arguments')
        exit(1)
    img_name = sys.argv[1]

    img = cv2.imread(img_name)

    if True:
        photo = PhotoCrop(img)
        crop = photo.get_correct_crop()

        is_fake = IsFakePhoto(crop)
        print(is_fake.has_magic_words(debug=True))
        print(is_fake.width_is_ok(debug=True))
        print(is_fake.color_is_ok(debug=True))
        is_fake.save_debuged_image()

    # except:
    #     print('not cropped')
    #     strange_words(img, True)
