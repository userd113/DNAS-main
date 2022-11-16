import tensorflow as tf
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Input,BatchNormalization
from keras.callbacks import EarlyStopping
import keras.losses
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class classfier():
    def __init__(self,
                 verbose=0,
                 seed=1234,
                 architecture=None
                 ):
        self.NN_parameters = architecture
        self.verbose = verbose
        self.seed = seed


    def predict(self,x,y):
        if y is not None:
            loss, accurancy = self.model.evaluate(x,y)
            print('test accurancy')
            print(accurancy)
        pred = self.model.predict(x)
        predict_y = pred.argmax(axis=1)
        return predict_y,pred
    def get_classifier(self):
        return self.model
    def fit(self,
            X_train,
            Y_train,
            X_test,
            Y_test
            ):
        inputdim=X_train.shape[1]
        model=self.build(inputdim)
        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, verbose=0,
                                                                 epsilon=0.0003, factor=0.9, min_lr=1e-6)
        result = model.fit(X_train, Y_train,
                           validation_data=(X_test, Y_test),
                           epochs=self.NN_parameters["epoch"],
                           batch_size=self.NN_parameters["batch_size"],
                           callbacks=[EarlyStopping(monitor='val_loss', patience=50, verbose=self.verbose),learning_rate_reduction],
                           verbose=0)#


        self.trained_epochs = len(result.history['loss'])
        print("Stopped fitting after {} epochs".format(self.trained_epochs))

        self.model=model
        loss, accurancy = model.evaluate(X_test, Y_test,verbose=0)
        return accurancy

    def build(self, inputdim):
        keras.backend.clear_session()
        input = Input(shape=(inputdim,))
        output = input

        for i in range(self.NN_parameters.get('nb_layers')):
            output =  Dense(self.NN_parameters['nb_neurons'][i], kernel_regularizer=keras.regularizers.l1_l2(l1=self.NN_parameters['l1_rate'][i],
                                                                                                       l2=self.NN_parameters['l2_rate'][i]), activation=self.NN_parameters['activation'][i])(output)

            if self.NN_parameters.get('dropout_rate')[i] != 0:
                output = Dropout(self.NN_parameters['dropout_rate'][i], seed=self.seed)(output)
            if self.NN_parameters.get('BN')[i] != 0:
                output = BatchNormalization()(output)
        output = Dense(4, activation='softmax')(output)


        model = Model(inputs=input, outputs=output)

        loss = self.NN_parameters['loss']
        #print(loss)
        if loss in [k for k, v in globals().items() if callable(v)]:
            # if loss is a defined function
            loss = eval(self.NN_parameters['loss'])

        if not callable(loss):
            # it is defined in Keras
            if hasattr(keras.losses, loss):
                loss = getattr(keras.losses, loss)
            else:
                print('Unknown loss: {}. Aborting.'.format(loss))
                exit(1)
        optim = self.get_optimizer()
        model.compile(optimizer=optim,
                      loss=loss,metrics=['categorical_accuracy'])
        return model

    def get_optimizer(self):
        if self.NN_parameters['optim'] == 'Adam':
            return keras.optimizers.Adam(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'SGD':
            return keras.optimizers.SGD(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adadelta':
            return keras.optimizers.Adadelta(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adagrad':
            return tf.keras.optimizers.Adagrad(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'RMSprop':
            return tf.keras.optimizers.RMSprop(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'NAdam':
            return tf.keras.optimizers.Nadam(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adamax':
            return tf.keras.optimizers.Adamax(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Ftrl':
            return tf.keras.optimizers.Ftrl(lr=self.NN_parameters['learning_rate'])
