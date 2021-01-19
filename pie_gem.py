import os, sys, time
import numpy as np
from pie_intent import PIEIntent
from pie_data import PIE
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from pedestrian_detection_ssdlite import api
from utils import *
from keras.applications import vgg16

# ffmpeg -r 3 -f image2 -s 1920x1440 -start_number 14 -i frame%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results.mp4

data_opts = {'fstride': 1,
        'sample_type': 'all',
        'height_rng': [0, float('inf')],
        'squarify_ratio': 0,
        'data_split_type': 'default',  #  kfold, random, default
        'seq_type': 'intention', #  crossing , intention
        'min_track_size': 0, #  discard tracks that are shorter
        'max_size_observe': 15,  # number of observation frames
        'max_size_predict': 5,  # number of prediction frames
        'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
        'balance': True,  # balance the training and testing samples
        'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
        'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
        'encoder_input_type': [],
        'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary']
        }

t = PIEIntent(num_hidden_units=128,
              regularizer_val=0.001,
              lstm_dropout=0.4,
              lstm_recurrent_dropout=0.2,
              convlstm_num_filters=64,
              convlstm_kernel_size=2)

pretrained_model_path = 'data/pie/intention/context_loc_pretrained'
data_path = './PIE_dataset'
img_path = './PIE_dataset/images'
set_id = 'set02'
vid_id = 'video_0003'
ped_id = '2_3_217'

#['2_3_194', '2_3_195', '2_3_196', '2_3_198', '2_3_199', '2_3_200', '2_3_201', '2_3_202', '2_3_203', '2_3_204', '2_3_205', '2_3_206', '2_3_207', '2_3_208', '2_3_209', '2_3_211', '2_3_212', '2_3_213', '2_3_214', '2_3_215', '2_3_216', '2_3_217', '2_3_210', '2_3_197']

imdb = PIE(data_path=data_path)
annt = imdb._get_annotations(set_id, vid_id)
ped = annt['ped_annotations']
ped_frames = ped[ped_id]['frames'][:30]
ped_frames = [i for i in range(63)]
#bbox = ped['ped_id']['bbox']
save_path = './bag1_results'

# 15 consecutive frames

# load bbox
bbox_sequences = ped[ped_id]['bbox'][:30]

VERSION = 'v0'
#Version v0, Total time: 31.7845561504364s, Per frame: 0.5045167642926413s
#Version v1, Total time: 24.467329025268555s, Per frame: 0.38837030198838973s

class Detector():
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        # test detector
        # image = cv2.imread(imp)
        # bbox_list = api.get_person_bbox(image, thr=0.6)
        # print(bbox_list)
        #
        # for i in bbox_list:
        #     cv2.rectangle(image, i[0], i[1], (125, 255, 51), thickness=2)
        #
        # plt.imshow(image[:, :, ::-1])
        # plt.show()

        # Detecting all the regions in the
        # Image that has a pedestrians inside it
        (regions, _) = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.1)

        # Drawing the regions in the Image
        rightmost = [0,0,0,0]
        for (x, y, w, h) in regions:
            #cv2.rectangle(image, (x, y),(x + w, y + h),(0, 0, 255), 2)
            if x+w > rightmost[2]:
                rightmost = [x,y,x+w,y+h]

        # shrink bbox
        x,y,xr,yr = rightmost
        w,h = xr-x, yr-y
        x,y = int(x + w/6), int(y+h/8)
        w,h = int(w*3/4), int(h*3/4)
        rightmost = [x,y,x+w,y+h]

        # Showing the output Image
        # cv2.rectangle(image,(rightmost[0], rightmost[1]), (rightmost[2], rightmost[3]) ,(0, 0, 255), 2)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()
        if rightmost == [0,0,0,0]:
            return None
        else:
            return rightmost

# load img
img_sequence = [] # 15 frames
feature_sequence = []
frameid_sequence = []
bbox_sequence = []
seq_length = 15
overlap = 0.5
step = 1#int(seq_length * overlap)
counter = 0

d = Detector()
t.load_pie_intent()
if VERSION == 'v1':
    convnet = vgg16.VGG16(input_shape=(224, 224, 3),
                                include_top=False,
                                weights='imagenet')

start = time.time()
for i, n in enumerate(ped_frames):
    #imp = os.path.join(img_path, set_id, vid_id, "%05d.png" % n)
    imp = os.path.join('./bag4_crossing', "frame%06d.png" % n)
    #imp = os.path.join('./bag3_playing_phone', "frame%06d.png" % n)
    #imp = os.path.join('./bag2_walking_no_crossing', "frame%06d.png" % n)
    #imp = os.path.join('./bag1_waiting', "frame%06d.png" % n)
    img = load_img(imp)

    image = cv2.imread(imp)
    bbox = d.detect(image)

    #bbox = bbox_sequences[i]
    # cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])) ,(0, 0, 255), 2)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if VERSION == 'v1':
        # preprocessing
        bbox1 = jitter_bbox_gem(img, [bbox[:]],'enlarge', 2)[0]
        #bbox = jitter_bbox_gem(img_data, [b],'same', 2)[0]
        bbox1 = squarify(bbox1, 1, img.size[0])
        bbox1 = list(map(int,bbox1[0:4]))
        cropped_image = img.crop(bbox1)
        img = img_pad(cropped_image, mode='pad_resize', size=224)
        image_array = img_to_array(img)
        preprocessed_img = vgg16.preprocess_input(image_array)
        expanded_img = np.expand_dims(preprocessed_img, axis=0)
        img_features = convnet.predict(expanded_img)
        img_features = np.squeeze(img_features) # (7,7,512)

    if bbox:
        img_sequence.append(img)
        if VERSION == 'v1':
            feature_sequence.append(img_features)
        frameid_sequence.append(n)
        bbox_sequence.append(bbox[:])
        if len(img_sequence) > 15:
            img_sequence = img_sequence[1:]
            if VERSION == 'v1':
                feature_sequence = feature_sequence[1:]
            frameid_sequence = frameid_sequence[1:]
            bbox_sequence = bbox_sequence[1:]

    counter += 1
    if counter >= step and len(img_sequence) == 15:
        counter = 0
        if VERSION == 'v0':
            ret = t.test_gem_v0(img_sequence, bbox_sequence)
        elif VERSION == 'v1':
            ret = t.test_gem_v1(feature_sequence, bbox_sequence)

        print("Intension probability: {} from frame {} to frame {}".format(ret[0], frameid_sequence[0], frameid_sequence[-1]))
        #cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])) ,(0, 0, 255), 2)
        #cv2.putText(image, 'Intension probability: {}'.format(ret[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        #cv2.imshow("Image", image)
        #cv2.imwrite(os.path.join(save_path, "frame%06d.png" % n), image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
end = time.time()
print("Version {}, Total time: {}s, Per frame: {}s".format(VERSION, end-start, (end-start)/len(ped_frames)))
