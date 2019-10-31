import warnings
import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib

# 设置log屏蔽info/warning/error, 只显示fatal信息
# https://blog.csdn.net/qq_40549291/article/details/85274581
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    

from keras import backend as K

# 以下两项设置效果相同
K.set_image_data_format('channels_first')   # 设置数据格式约定的值
# K.set_image_dim_ordering('th')              # 设置图片通道顺序

# 模型的文件路径
# CNN(VGG16)的权重文件: https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
VQA_model_file_name = 'models/VQA/VQA_MODEL.json'
VQA_weights_file_name = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name = 'models/CNN/vgg16_weights.h5'

# verbose = 0 不输出中间过程状态
verbose = 1

'''
由输入的CNN权重文件(vgg16_weights.h5)返回带权的VGG模型
需要models/CNN/VGG.py
'''
def get_image_model(CNN_weights_file_name):    
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)

    # 标准VGG16网络, 去掉最后两层
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # optimizer可以尝试用adam, 但是这种任务的损失函数非常标准
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return image_model

''' 
Runs the given image_file to VGG 16 model and returns the weights (filters) as a 1, 4096 dimension vector
把给定的image_file输入VGG_16模型, 返回权重(filters)是一个(1,4096)的向量
'''
def get_image_features(image_file_name, CNN_weights_file_name):
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image 
    # is required to go through the same transformation
    # 因为VGG把训练图像作为224x224, 每张新的图像都要resize成相同尺寸
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))

    # 平均像素值来自VGG的作者, 由训练数据集计算得出
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    # type(im): np.ndarray
    # im.shape: (224, 224, 3)

    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    im = im.transpose((2, 0, 1))  # convert the image to RGBA
    # im.shape: (3, 224, 224)

    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size)
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)
    # im.shape: (1, 3, 224, 224)

    vgg_model = get_image_model(CNN_weights_file_name)

    # from keras.utils.visualize_util import plot
    from keras.utils.vis_utils import plot_model
    plot_model(vgg_model, to_file='model_vgg.json')

    image_features[0, :] = vgg_model.predict(im)[0]
    return image_features

'''
Given the VQA model and its weights, compiles and returns the model
'''
def get_VQA_model(VQA_weights_file_name):
    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

    # 试图从VQA_MODEL.json中加载出VQA的模型, 但未成功
    # vqa_model = model_from_json(open(VQA_model_file_name).read())
    # vqa_model.load_weights(VQA_weights_file_name)
    # vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # return vqa_model

''' 
For a given question, a unicode string, returns the timeseris vector
with each word (token) transformed into a 300 dimension representation
calculated using Glove Vector
对于给定的问题，一个Unicode字符串返回timeseris向量, 
其中每个word(token)转换为使用Glove Vector计算的300维表示
'''
def get_question_features(question):
    # 这几行待学习NLP......
    # word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0, j, :] = tokens[j].vector
    return question_tensor

''' 
accepts command line arguments for image file and the question and
builds the image model (VGG) and the VQA model (LSTM and MLP)
prints the top 5 response along with the probability of each
从命令行中接收image file和question的参数, 并建立image model(VGG)和VQA model(LSTM和MLP)
根据每个的概率打印出top 5的回答 
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='test.jpg')

    # 这两个问题意思相同, 但准确率差了很多, 有待研究......
    # parser.add_argument('-question', type=str, default='What vechile is in the picture?')
    parser.add_argument('-question', type=str, default='What is the vechile in the picture?')

    args = parser.parse_args()

    if verbose: print("\n\n\nLoading image features ...")
    image_features = get_image_features(args.image_file_name, CNN_weights_file_name)

    if verbose: print("\n\n\nLoading question features ...")
    question_features = get_question_features(args.question)

    if verbose: print("\n\n\nLoading VQA Model ...")
    vqa_model = get_VQA_model(VQA_weights_file_name)

    if verbose: print("\n\n\nPredicting result ...")
    y_output = vqa_model.predict([question_features, image_features])

    # 返回对y_output数组进行排序的索引
    y_sort_index = np.argsort(y_output)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    # 此任务在此表示为对1000个最佳答案的分类, 这意味着某些答案不是训练的一部分, 因此不会在结果中显示
    # 这1000个答案存储在sklearn Encoder类中
    labelencoder = joblib.load(label_encoder_file_name) # 存储了1000个最佳答案
    for label in reversed(y_sort_index[0, -5:]):        # 输出可能性最大的五个答案
        l = []
        l.append(label)
        print(str(round(y_output[0, label] * 100, 2)).zfill(5), "% ", labelencoder.inverse_transform(l))

if __name__ == "__main__":
    main()