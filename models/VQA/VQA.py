# modification of model from https://github.com/avisingh599/visual-qa
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Concatenate, concatenate, Dense
from keras.layers import Merge

def VQA_MODEL():
    image_feature_size          = 4096
    word_feature_size           = 300
    number_of_LSTM              = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions        = 30
    number_of_dense_layers      = 3
    number_of_hidden_units      = 1024
    activation_function         = 'tanh'
    dropout_pct                 = 0.5

    # Image model
    model_image = Sequential()
    model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

    # Language Model
    model_language = Sequential()
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    print("\n\n\nmodel_image_summary:")
    print(model_image.summary())
    # print(model_language.summary())

    # type(model_language): <class 'keras.engine.sequential.Sequential'>
    # type(model_image): <class 'keras.engine.sequential.Sequential'>

    model = Sequential()    # Combined Model
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))
    # 在版本不同时会报错, 下面是试图改正错误的代码, 但未成功
    # model.add(Concatenate(axis=1)([model_language, model_image]))
    # Concatenate([model_language, model_image], axis=1)

    # print(model.summary())

    for _ in range(number_of_dense_layers):
        model.add(Dense(number_of_hidden_units, kernel_initializer='uniform'))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_pct))

    model.add(Dense(1000))
    model.add(Activation('softmax'))

    return model