import keras
import warnings
import numpy as np
from preprocess import load_data
from keras.models import Sequential
from myLayers.attention import MultiHeadAttention
from keras.layers import Bidirectional,LSTM,Dense,Activation,Conv1D,GlobalMaxPool1D,Input,Flatten,Dropout

warnings.filterwarnings('ignore')

class LstmEncoderDecoder():

    def __init__(self,window_size,file_train,file_test,split,batch_size,learning_rate,epoch,dataSave=False):
        self.model = 'LSTM-Encoder-Decoder'
        self.window_size = window_size
        self.metric = 'mse'
        self.file_train = file_train
        self.file_test = file_test
        self.split =split
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.config = {'LSTM_unit1':50,
                       'LSTM_unit2':100,
                       'Dense_unit':1}
        print('>>> DataPrecessing')
        self.train_x, self.train_y = load_data(self.file_train,
                                                           self.window_size,
                                                           self.split,
                                                            train=True)

        self.test_x, self.test_y = load_data(self.file_test,
                                                         self.window_size,
                                                         self.split,
                                                            train=False)
        if dataSave:
            self.ndarrayData()


    def ndarrayData(self):
        np.save('./npData/train_x.npy',self.train_x)
        np.save('./npData/train_y.npy',self.train_y)
        np.save('./npData/test_x.npy',self.test_x)
        np.save('./npData/test_y.npy',self.test_y)

    def _create_model(self):
        model = Sequential()
        model.add(LSTM(input_dim=1, output_dim=self.config['LSTM_unit1'], return_sequences=True))
        model.add(LSTM(self.config['LSTM_unit2'], return_sequences=False))
        model.add(Dense(output_dim=self.config['Dense_unit']))
        model.add(Dropout(rate=0.2))
        model.add(Activation('linear'))
        model.compile(loss=self.metric,
                      optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9,
                                                      beta_2=0.999, epsilon=None,
                                                      decay=0.0, amsgrad=False))

        print(model.summary())
        return model

    def fit(self):
        model = self._create_model()
        model.fit(x=self.train_x,
                  y=self.train_y,
                  epochs = self.epoch,
                  batch_size = self.batch_size,
                  validation_data = (self.test_x,self.test_y))
        model.save('./modelCheckPoint/EncDec.h5')

    def predict(self):
        model = keras.models.load_model('./modelCheckPoint/EncDec.h5')
        y_hat = model.predict(self.test_x,batch_size = self.batch_size)
        np.save('./result/lstmEncDnc_yHat.npy',y_hat)
        return y_hat

class Conv1DAutoEncoder():
    def __init__(self,window_size,file_train,file_test,split,batch_size,learning_rate,epoch,dataSave=False):
        self.model = 'Conv1DAutoEncoder'
        self.window_size = window_size
        self.metric = 'mse'
        self.file_train = file_train
        self.file_test = file_test
        self.split =split
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.config = {'filters':256,
                       'kernel_size':7,
                       'conv_activation':'relu',
                       'padding':'same',
                       'Dense_unit':1}
        print('>>> DataPrecessing')
        self.train_x, self.train_y = load_data(self.file_train,
                                                           self.window_size,
                                                           self.split,
                                                            train=True)

        self.test_x, self.test_y = load_data(self.file_test,
                                                         self.window_size,
                                                         self.split,
                                                            train=False)
        if dataSave:
            self.ndarrayData()

    def ndarrayData(self):
        np.save('./npData/train_x.npy',self.train_x)
        np.save('./npData/train_y.npy',self.train_y)
        np.save('./npData/test_x.npy',self.test_x)
        np.save('./npData/test_y.npy',self.test_y)

    def _create_model(self):
        model = Sequential()
        model.add(Conv1D(filters = self.config['filters'],
                         kernel_size = self.config['kernel_size'],
                         activation = self.config['conv_activation'],
                         input_dim = 1,
                         padding = self.config['padding'])
                  )
        model.add(GlobalMaxPool1D())
        model.add(Dense(output_dim=self.config['Dense_unit']))
        model.add(Dropout(rate = 0.2))
        model.add(Activation('linear'))
        model.compile(loss=self.metric,
                      optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9,
                                                      beta_2=0.999, epsilon=None,
                                                      decay=0.0, amsgrad=False))
        return model

    def fit(self):
        model = self._create_model()
        model.fit(x=self.train_x,
                  y=self.train_y,
                  epochs = self.epoch,
                  batch_size = self.batch_size,
                  validation_data = (self.test_x,self.test_y))
        model.save('./modelCheckPoint/Conv1DAutoEncoder.h5')

    def predict(self):
        model = keras.models.load_model('./modelCheckPoint/Conv1DAutoEncoder.h5')
        y_hat = model.predict(self.test_x,batch_size = self.batch_size)
        np.save('./result/Conv1DAutoEncoder_yHat.npy',y_hat)
        return y_hat

class Conv1DLSTM():
    def __init__(self,window_size,file_train,file_test,split,batch_size,learning_rate,epoch,dataSave=False):
        self.model = 'Conv1DLSTM'
        self.window_size = window_size
        self.metric = 'mse'
        self.file_train = file_train
        self.file_test = file_test
        self.split =split
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.config = {'filters':256,
                       'kernel_size':7,
                       'conv_activation':'relu',
                       'padding':'same',
                       'LSTM_unit':50,
                       'Dense_unit':1}
        print('>>> DataPrecessing')
        self.train_x, self.train_y = load_data(self.file_train,
                                                           self.window_size,
                                                           self.split,
                                                            train=True)

        self.test_x, self.test_y = load_data(self.file_test,
                                                         self.window_size,
                                                         self.split,
                                                            train=False)
        if dataSave:
            self.ndarrayData()


    def ndarrayData(self):
        np.save('./npData/train_x.npy',self.train_x)
        np.save('./npData/train_y.npy',self.train_y)
        np.save('./npData/test_x.npy',self.test_x)
        np.save('./npData/test_y.npy',self.test_y)

    def _create_model(self):
        model = Sequential()
        model.add(Conv1D(filters = self.config['filters'],
                         kernel_size = self.config['kernel_size'],
                         activation = self.config['conv_activation'],
                         input_dim = 1,
                         padding = self.config['padding'])
                  )
        model.add(Dropout(rate = 0.2))
        model.add(LSTM(self.config['LSTM_unit'], return_sequences=False))
        model.add(Dropout(rate = 0.2))
        model.add(Dense(output_dim=self.config['Dense_unit']))
        model.add(Dropout(rate = 0.2))
        model.add(Activation('linear'))
        model.compile(loss=self.metric,
                      optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9,
                                                      beta_2=0.999, epsilon=None,
                                                      decay=0.0, amsgrad=False))
        return model

    def fit(self):
        model = self._create_model()
        model.fit(x=self.train_x,
                  y=self.train_y,
                  epochs = self.epoch,
                  batch_size = self.batch_size,
                  validation_data = (self.test_x,self.test_y))
        model.save('./modelCheckPoint/Conv1DLSTM.h5')

    def predict(self):
        model = keras.models.load_model('./modelCheckPoint/Conv1DLSTM.h5')
        y_hat = model.predict(self.test_x,batch_size = self.batch_size)
        np.save('./result/conv1DLSTM_yHat.npy', y_hat)
        return y_hat


class AttLSTM():
    def __init__(self, window_size, file_train, file_test, split, batch_size, learning_rate, epoch, dataSave=False):
        self.model = 'AttentionLSTM'
        self.window_size = window_size
        self.metric = 'mse'
        self.file_train = file_train
        self.file_test = file_test
        self.split = split
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.config = {'LSTM_unit1': 50,
                       'LSTM_unit2': 1}
        print('>>> DataPrecessing')
        self.train_x, self.train_y = load_data(self.file_train,
                                                           self.window_size,
                                                           self.split,
                                                            train=True)

        self.test_x, self.test_y = load_data(self.file_test,
                                                         self.window_size,
                                                         self.split,
                                                            train=False)
        if dataSave:
            self.ndarrayData()

    def ndarrayData(self):
        np.save('./npData/train_x.npy', self.train_x)
        np.save('./npData/train_y.npy', self.train_y)
        np.save('./npData/test_x.npy', self.test_x)
        np.save('./npData/test_y.npy', self.test_y)

    def _create_model(self):
        i = Input(shape=(self.window_size, 1))
        enc = LSTM(self.config['LSTM_unit1'], return_sequences=True)(i)
        dec,attn = MultiHeadAttention(2,100,0.2,1)(enc,enc,enc)
        result = LSTM(units=self.config['LSTM_unit2'], activation='linear')(dec)
        model = keras.Model(inputs=i, outputs=result)
        model.compile(loss=self.metric,
                      optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9,
                                                      beta_2=0.999, epsilon=None,
                                                      decay=0.0, amsgrad=False))
        return model

    def fit(self):
        model = self._create_model()
        model.fit(x=self.train_x,
                  y=self.train_y,
                  epochs=self.epoch,
                  batch_size=self.batch_size,
                  validation_data=(self.test_x, self.test_y))


        return model

    def predict(self):
        model = self.fit()
        y_hat = model.predict(self.test_x, batch_size=self.batch_size)
        np.save('./result/AttnBi-LSTM.npy', y_hat)
        return y_hat

if __name__ == '__main__':

    MODEL = 'Conv1DLSTM'
    if MODEL == 'LSTM_enc_dec':
        LSTM_enc_dec = LstmEncoderDecoder(window_size=50,
                                          file_train = './data/nyc_taxi_train.csv',
                                          file_test='./data/nyc_taxi_test.csv',
                                          split = 1.,
                                          batch_size = 32,
                                          learning_rate=.001,
                                          epoch = 20,
                                          dataSave = True)
        LSTM_enc_dec.fit()
        LSTM_enc_dec.predict()


    elif MODEL == 'Conv1D':
        conv1D = Conv1DAutoEncoder(window_size=50,
                                   file_train='./data/nyc_taxi_train.csv',
                                   file_test='./data/nyc_taxi_test.csv',
                                    split = 1.,
                                    batch_size = 32,
                                    learning_rate=.001,
                                    epoch = 20,
                                    dataSave = False)
        conv1D.fit()
        conv1D.predict()

    elif MODEL == 'Conv1DLSTM':
        conv1dLSTM = Conv1DLSTM(window_size=50,
                                file_train='./data/nyc_taxi_train.csv',
                                file_test='./data/nyc_taxi_test.csv',
                                split = 1.,
                                batch_size = 32,
                                learning_rate=.001,
                                epoch = 20,
                                dataSave = False)
        conv1dLSTM.fit()
        conv1dLSTM.predict()

    elif MODEL == 'AttentionLSTM':
        attLSTM = AttLSTM(window_size=50,
                          file_train='./data/nyc_taxi_train.csv',
                          file_test='./data/nyc_taxi_test.csv',
                          split = 1,
                          batch_size = 32,
                          learning_rate=.015,
                          epoch = 20,
                          dataSave = False)
        attLSTM.predict()


