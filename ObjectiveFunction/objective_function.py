from ObjectiveFunction import ObjectiveBase

#from tensorflow.python.keras.api.keras.preprocessing import image
from keras.api.keras.preprocessing import image

from NeuralNetwork.utils.image_enhancer import ImageEnhancer
from NeuralNetwork.utils.image_preprocessing import ImagePreprocessing
from NeuralNetwork.model import Nima
import os
import random

wheights = [20, 10, -4, 1]


def prepare_model():
    nima = Nima()
    nima.build()

    if True:
        nima.nima_model.load_weights("./NeuralNetwork/weights/weights_mobilenet_aesthetic_0.07.hdf5")

    return nima


def bin_to_float(binary):
    # print('binary',str(binary))
    new_seed = "".join(str(x) for x in binary)
    # print('c2', new_seed)
    # print('c2', int(new_seed,2))
    random.seed(int(new_seed,2))
    value = random.uniform(0.7, 3)
    return value


class ObjectiveFunction(ObjectiveBase):
    """
    A class of Objective function.
    """

    def __init__(self, dim, eda_algorithm, lam, img_directory, image_id, minimize=False):
        super(ObjectiveFunction, self).__init__(dim, eda_algorithm, lam, img_directory, image_id, minimize=minimize)
        self.optimal_value = 0 if minimize else 1

    nima = prepare_model()

    def evaluate(self, c):
        # self.lam = 2
        img_path = os.path.join(self.img_directory, str(self.image_id) + ".jpg")
        img = image.load_img(img_path, target_size=(224, 224))

        c = self._check_shape(c)
        #print("Final C = ", len(c))

        dict = {'contrast': 2.0, 'brightness': 6.9, 'sharpness': 6.9, 'color': 6.9}
        # score = [0]*len(c)  --> CGA  --> en range tambien
        score = [0] * self.lam
        interval = self.lam
        if self.eda_algorithm == "ecga" or self.eda_algorithm == "cga":
            #print("OK")
            interval = len(c)
            score = [0] * len(c)

        for i in range(interval):
            # print(c[i][0:10])
            dict['contrast'] = bin_to_float(c[i][0:10])
            dict['brightness'] = bin_to_float(c[i][10:20])
            dict['sharpness'] = bin_to_float(c[i][20:30])
            dict['color'] = bin_to_float(c[i][30:40])

            enhancer = ImageEnhancer(img, dict['contrast'], dict['brightness'], dict['sharpness'],
                                     dict['color'])
            enhanced_image = enhancer.enhance_image()
            preprocessor = ImagePreprocessing(enhanced_image)
            image_ready = preprocessor.prepare_for_model()

            score[i] = 1 - self.nima.predict(image_ready, mean=True)

            # score[i] = wheights[0] * dict['contrast'] ** 2 + wheights[1] * (np.pi * dict['brightness']) - 2 - \
            #            wheights[2] * dict['sharpness'] + wheights[3] * dict['color']

        # print("Score: ",score)
        # self.lam = int(self.lam * 0.5)
        # print(self.lam)
        # print(score)
        # if save:
        #     scores = nima.predict(image_ready, mean=False)
        #     scores = scores[0].tolist()
        #     scores.insert(0, image_id)
        #     scores.insert(1, original)
        #     return before_after.append(scores)

        # c = self._check_shape(c)
        # print(c)
        # evals = np.sum(c, axis=1)
        # print(evals)
        #
        # evals = -evals if self.minimize else evals
        info = {}
        #
        return score, info

    def __str__(self):
        sup_str = "  " + super(ObjectiveFunction, self).__str__().replace("\n", "  ")
        return 'ObjectiveFunction(''{}'')'.format(sup_str)
