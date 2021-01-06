#! /usr/bin/python

import sys
from networkconfig import backend_config as c

print("Importing Keras packages")
from keras.layers.convolutional import Conv2D
from keras.layers import Flatten, MaxPooling2D, AveragePooling2D, Lambda, TimeDistributed, Dense, Dropout, ConvLSTM2D, LSTM, AveragePooling2D, concatenate
from keras.models import load_model, Input
from keras.models import Model
from keras.utils import to_categorical
from keras import regularizers
import tensorflow as tf

# Generic Imports
print("Importing generic Packs")
import os
import numpy as np
# np.set_printoptions(threshold=np.nan)
from tqdm import tqdm
import cv2
from enum import Enum

###### Uncomment if you want to use train on particular GPU ########
# gpu = input("Enter GPU number : ")
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu


class ACT(Enum):
    InspectAfterPicking = 0 
    PlaceOnConveyor = 1 
    PlaceInBin = 2 
    Pick = 3 
    ClaimNewOnion = 4 
    InspectWithoutPicking = 5 
    ClaimNextInList = 6


class nnn:
    # Constants will be here
    print("Init SA-Net")
    # Data Parameters
    INFO = True
    configs = c.configs_raw()

    # data arrays
    dataset = np.array([])  # will contain the images/Videos
    dataset_o = np.array([])  # will contain the images/Videos
    rgbpfiles = {}  # the key value pair for data images
    depthpfiles = {}  # the key value pair for data depth images
    actionlables = []  # the key value pair for action labels
    statelables = []  # the key value pair for state labels
    statelables_x = []  # the key value pair for state labels x
    statelables_y = []  # the key value pair for state labels y
    statelables_o = []  # the key value pair for state labels o -> Theta
    statelables_local = []  # the key value pair for state label localization
    relativeY = []  # Relative location of the robot X
    relativeX = []  # Relative location of the robot Y
    statexycoord = []  # Debug only to store the XY coordnates of location of the bot
    finallables = []

    # counters (files,images etc)
    rgbfileset = 0
    depthfileset = 0

    # Paths
    datapath = configs.datapath
    model_path = configs.model_path
    model_path_kfold = configs.model_path_kfold
    datapath_test = configs.datapath_test
    kfolds_train_path = configs.kfolds_train_path
    kfolds_test_path = configs.kfolds_test_path
    logs_dir = configs.logs_dir


    ###### Uncomment if you want to use fractional GPU Memory ########
    # from keras.backend.tensorflow_backend import set_session
    # config = tf.ConfigProto()
    # #config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # sess = tf.Session(config=config)
    # set_session(sess)

    model = None

    # Height x Width x Channel ||| Here Channel = Red,Green,Blue,Depth
    INPUT_SHAPE_STATE = configs.INPUT_SHAPE_STATE

    # Samples x Height x Width x Channel ||| Here Channel = Red,Green,Blue,Depth
    INPUT_SHAPE_TIME = configs.INPUT_SHAPE_TIME
    INPUT_SHAPE_TIME_ACTION = configs.INPUT_SHAPE_TIME_ACTION

    # Network Hyper-Parameters
    BATCH_SIZE = configs.BATCH_SIZE
    BATCH = []
    TOTAL_EPOCHS = configs.TOTAL_EPOCHS
    kfolds = configs.kfolds
    EPOCHS = configs.EPOCHS

    # RNN Parameters
    TIME_GAP = configs.TIME_GAP  # No of frames that has to be dropped between t,t-1,t-2

    def __init__(self):
        import sys
        print("Inititilizing SA-NET Network")
        if os.path.exists(self.model_path) == True:
            if sys.version_info >= (3, 0):
                self.model = load_model(self.model_path)
            else:
                self.create_graph_n_load()
        else:
            print("Could not find trained model\n Recommended course of action:\n ->Train first then try")

        if os.path.exists(self.configs.checkpoint_folder) == False:
            print("Can't find checkpoints folder, Creating one now")
            os.mkdir(self.configs.checkpoint_folder)

    def printshapes(self):
        print("X :", self.statelables_x.shape)
        print("Y :", self.statelables_y.shape)
        print("O :", self.statelables_o.shape)
        print("local :", self.statelables_local.shape)

    def writetofile(self, info):
        with open('info', 'wb') as f:
            f.write(info)

    def getNetworkInformation(self):
        if self.INFO == True:
            self.model.summary()
        else:
            print("Model Summary is temporarily disabled")

    

    def get_numpy(self):
        '''
        Converts all Data to a numpy array and save into file which can be taken in if its already there without crop
        <Depricated>
        :return: Nothing
        '''
        filepath = "data/numpydata/actionrd.npy"
        if os.path.exists(filepath) == False:
            temp = list()
            print("Numpy File does not exist importing data")
            for index in range(len(self.rgbpfiles)):
                # Get all RGB data for one set ch:1,2,3
                image_t = cv2.imread(self.rgbpfiles[index][0])
                image_t1 = cv2.imread(self.rgbpfiles[index][1])
                image_t2 = cv2.imread(self.rgbpfiles[index][2])
                r_image_t = cv2.resize(image_t, (224, 224), interpolation=cv2.INTER_CUBIC)
                r_image_t1 = cv2.resize(image_t1, (224, 224), interpolation=cv2.INTER_CUBIC)
                r_image_t2 = cv2.resize(image_t2, (224, 224), interpolation=cv2.INTER_CUBIC)

                # Get all Depth Data for one set ch : 4
                image_t_d = cv2.imread(self.depthpfiles[index][0], 0)
                image_t1_d = cv2.imread(self.depthpfiles[index][1], 0)
                image_t2_d = cv2.imread(self.depthpfiles[index][2], 0)
                d_image_t = cv2.resize(image_t_d, (224, 224), interpolation=cv2.INTER_CUBIC)
                d_image_t1 = cv2.resize(image_t1_d, (224, 224), interpolation=cv2.INTER_CUBIC)
                d_image_t2 = cv2.resize(image_t2_d, (224, 224), interpolation=cv2.INTER_CUBIC)
                d_image_t = d_image_t[:, :, None]
                d_image_t1 = d_image_t1[:, :, None]
                d_image_t2 = d_image_t2[:, :, None]

                # concat and add to create a 4 channel system
                packet = np.array(
                    [np.concatenate((r_image_t, d_image_t), axis=2), np.concatenate((r_image_t1, d_image_t1), axis=2),
                     np.concatenate((r_image_t2, d_image_t2), axis=2)])
                temp.append(packet)

            # Add to dataset
            self.dataset = np.array(temp)
            # print(self.dataset.shape)
            np.save("data/numpydata/actionrd", self.dataset)


        else:
            print("Numpy File exists importing data")
            self.dataset = np.load(filepath)

    def imnormalize(self, image):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : image:image data.
        : return: Numpy array of normalize data
        """
        xmin = 0
        xmax = 255
        a = 0
        b = 1
        return ((image - xmin) * (b - a)) / (xmax - xmin)

    

    def initData(self, datapath):

        filepath_o = datapath + "dataset_o_" + str(self.TIME_GAP) + ".npy"
        filepath_s = datapath + "dataset_s_" + str(self.TIME_GAP) + ".npy"

        if os.path.exists(filepath_o) == False:
            print(filepath_o + " FAIL")
        else:
            print(filepath_o + " PASS")

        if os.path.exists(filepath_o) == False:
            print(filepath_s + " FAIL")
        else:
            print(filepath_s + " PASS")

        if os.path.exists(filepath_o) == False or os.path.exists(filepath_s) == False:
            print("Error Init of data. DATA failed")
            exit(0)
        else:
            print("Numpy File exists importing data")
            self.dataset_o = np.load(filepath_o)
            self.dataset_s = np.load(filepath_s)

    def initLabels(self, labpath):

        actionfilepath = labpath + "actionlables_" + str(self.TIME_GAP) + ".npy"
        statefilepath_x = labpath + "statelables_x_" + str(self.TIME_GAP) + ".npy"
        statefilepath_y = labpath + "statelables_y_" + str(self.TIME_GAP) + ".npy"
        statefilepath_o = labpath + "statelables_o_" + str(self.TIME_GAP) + ".npy"

        if os.path.exists(actionfilepath) == False:
            print(actionfilepath + " FAIL")
        else:
            print(actionfilepath + " PASS")

        if os.path.exists(statefilepath_x) == False:
            print(statefilepath_x + " FAIL")
        else:
            print(statefilepath_x + " PASS")

        if os.path.exists(statefilepath_y) == False:
            print(statefilepath_y + " FAIL")
        else:
            print(statefilepath_y + " PASS")

        if os.path.exists(statefilepath_o) == False:
            print(statefilepath_o + " FAIL")
        else:
            print(statefilepath_o + " PASS")

        print()
        if os.path.exists(actionfilepath) == False or os.path.exists(statefilepath_x) == False or os.path.exists(
                statefilepath_y) == False or os.path.exists(statefilepath_o) == False:
            print("Error Init of labels. LABELS failed")
            exit(0)

        else:
            print("State and Action Lables Found Importing")
            self.actionlables = np.load(actionfilepath)
            self.statelables_x = np.load(statefilepath_x)
            self.statelables_y = np.load(statefilepath_y)
            self.statelables_o = np.load(statefilepath_o)


    # Function to create callbacks for keras can be added here
    def callback_create(self):
        '''

        :return:
        '''
        import keras
        callbacks = []

        # For every epoch a snapshot is created and can be tested how the network works
        chkpt="checkpoints/NNN"+str(self.kfolds)+"_{epoch:02d}.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(chkpt, verbose=1,
                                                     save_best_only=True)
        callbacks.append(checkpoint)

        # Tensorboard for tracking tensors
        tensor_board = keras.callbacks.TensorBoard(log_dir=self.logs_dir, histogram_freq=0, batch_size=self.BATCH_SIZE,
                                                   write_graph=True, write_grads=True, write_images=True,
                                                   embeddings_freq=0, embeddings_layer_names=None,
                                                   embeddings_metadata=None)
        callbacks.append(tensor_board)

        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='Action_loss',
            factor=0.01,
            patience=2,
            verbose=1,
            mode='auto',
            epsilon=0.0001,
            cooldown=0,
            min_lr=0
        ))

        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       min_delta=0,
                                                       patience=5,
                                                       verbose=1, mode='auto'))

        callbacks.append(keras.callbacks.CSVLogger("logcsv.csv", separator=',', append=False))
        return callbacks

    # CNN part of the data
    def create_cnn_model_time_fn(self):
        input = Input(shape=self.INPUT_SHAPE_STATE, name="Input_State_X_Y")

        cmodel_layer1 = Conv2D(32, kernel_size=(3, 12), strides=(3, 3), activation='relu', name="cnn_state_XY_layer1")(input)
        cmodel_layer1_maxpool = AveragePooling2D(pool_size=(4, 4), strides=(3, 3), name="pool_state_XY_node1")(cmodel_layer1)

        cmodel_layer2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="cnn_state_XY_layer_2")(cmodel_layer1_maxpool)
        cmodel_layer3 = Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), name="cnn_state_XY_layer3")(cmodel_layer2)
        cmodel_layer3_maxpool = AveragePooling2D(pool_size=(2, 2), strides=(3, 3), name="pool_state_XY_node2")(cmodel_layer3)

        cmodel_layer4 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="cnn_state_XY_layer4")(cmodel_layer3_maxpool)
        cmodel_layer5 = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), name="cnn_state_XY_layer5")(cmodel_layer4)
        cmodel_layer5_maxpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool_state_XY_node3")(cmodel_layer5)

        cmodel = Flatten()(cmodel_layer5_maxpool)
        return cmodel, input

    def create_cnn_model_time_fn_action(self):
        input = Input(shape=self.INPUT_SHAPE_TIME_ACTION, name="Input_Action_and_Theta")

        cmodel_layer1 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'), name="cnn_action_layer1")(input)

        cmodel_layer3 = TimeDistributed(
            Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_regularizer=regularizers.l2(0.01)), name="cnn_action_layer3")(cmodel_layer1)
        cmodel_layer3_maxpool = TimeDistributed(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))(cmodel_layer3)

        return cmodel_layer3_maxpool, input

    def o_model(self, x, x2):

        # To get the first dimention of the tesnor and pass it to the next layer

        o_layer1 = TimeDistributed(Conv2D(128, kernel_size=(2, 2), strides=(2, 2), activation='relu'), name="cnn_O_layer1")(x)
        flatten_time = TimeDistributed(Flatten())(o_layer1)
        flatten = Flatten()(flatten_time)
        merge_state_theta = concatenate([flatten, x2])

        statemodel_o = Dense(5, activation="softmax", name="State_o")(merge_state_theta)  # 0<=o<=3  +1 nobot state
        return statemodel_o ,merge_state_theta

    # RNN-LSTM action part of it
    def rnn_part_timefn(self, rmodel,xytheta_model):

        rmodel_convLSTM_layer1 = ConvLSTM2D(filters=20, kernel_size=(3, 3), strides=(2, 2), return_sequences=True, kernel_regularizer=regularizers.l2(0.01), dropout=0.5,
                                            activation='relu', name="Action_ConvLSTM_Layer1")(rmodel)

        rmodel_convLSTM_layer2 = ConvLSTM2D(filters=5, kernel_size=(2, 2), strides=(3, 3), return_sequences=False, kernel_regularizer=regularizers.l2(0.01), dropout=0.5,
                                            activation='relu', name="Action_ConvLSTM_Layer2")(rmodel_convLSTM_layer1)

        flatten = concatenate([Flatten()(rmodel_convLSTM_layer2),xytheta_model])

        rmodel_output = Dense(5, activation="softmax", name="Action")(flatten)  # 0<=action<=3  +1 nobot state

        return rmodel_output

    def rnn_part_timefn_arm(self, rmodel,xytheta_model):

        rmodel_convLSTM_layer1 = ConvLSTM2D(filters=20, kernel_size=(3, 3), strides=(2, 2), return_sequences=True, kernel_regularizer=regularizers.l2(0.01), dropout=0.5,
                                            activation='relu', name="Action_ConvLSTM_Layer1")(rmodel)

        rmodel_convLSTM_layer2 = ConvLSTM2D(filters=5, kernel_size=(2, 2), strides=(3, 3), return_sequences=False, kernel_regularizer=regularizers.l2(0.01), dropout=0.5,
                                            activation='relu', name="Action_ConvLSTM_Layer2")(rmodel_convLSTM_layer1)

        flatten = concatenate([Flatten()(rmodel_convLSTM_layer2),xytheta_model])

        rmodel_output = Dense(5, activation="softmax", name="Action")(flatten)  # 0<=action<=3  +1 nobot state

        return rmodel_output

    def localizationmodule(self, x):

        state_local1 = Dense(75, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_local1")(x)
        state_local1_do = Dropout(0.5)(state_local1)

        state_local2 = Dense(25, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_local2")(state_local1_do)
        state_local2_do = Dropout(0.5)(state_local2)

        state_local_output = Dense(2, activation="softmax", name="state_local_output")(state_local2_do)

        return state_local2_do, state_local_output

    def relativemodel(self, x):

        state_relative1 = Dense(75, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="relative_layer_1")(x)
        state_relative1_do = Dropout(0.5)(state_relative1)

        state_relative2 = Dense(75, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="relative_layer_2")(state_relative1_do)
        state_relative2_do = Dropout(0.5)(state_relative2)

        relative_x_out = Dense(3, activation="softmax", name="Relative_x")(state_relative2_do)  # 0<=x<=3  +1 nobot state
        relative_y_out = Dense(18, activation="softmax", name="Relative_y")(state_relative2_do)  # 0<=y<=16 +1 nobot state

        return relative_x_out, relative_y_out, state_relative2_do

    def relativemodel_arm(self, x):

        state_relative1 = Dense(75, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="relative_layer_1")(x)
        state_relative1_do = Dropout(0.5)(state_relative1)

        state_relative2 = Dense(75, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="relative_layer_2")(state_relative1_do)
        state_relative2_do = Dropout(0.5)(state_relative2)

        relative_x_out = Dense(10, activation="softmax", name="Relative_x")(state_relative2_do)  # 0<=x<=8  +1 nobot state
        relative_z_out = Dense(18, activation="softmax", name="Relative_y")(state_relative2_do)  # 0<=y<=16 +1 nobot state
        relative_y_out = Dense(18, activation="softmax", name="Relative_z")(state_relative2_do)  # 0<=y<=16 +1 nobot state

        return relative_x_out, relative_y_out,relative_z_out, state_relative2_do

    def statemodel(self, x):

        relative_x_out, relative_y_out, relativemod = self.relativemodel(x)

        state_layer1 = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer1")(x)
        state_layer1_do = Dropout(0.5)(state_layer1)
        merge_state_local_layer = concatenate([state_layer1_do, relativemod])
        state_layer2 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer2")(merge_state_local_layer)
        state_layer2_do = Dropout(0.5)(state_layer2)

        merge_state_local_layer = concatenate([state_layer2_do, relativemod])  # layer merges here

        state_layer3 = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer3")(merge_state_local_layer)
        state_layer3_do = Dropout(0.5)(state_layer3)

        statemodel_x = Dense(3, activation="softmax", name="State_x")(state_layer3_do)  # 0<=x<=3  +1 nobot state
        statemodel_y = Dense(18, activation="softmax", name="State_y")(state_layer3_do)  # 0<=y<=16 +1 nobot state

        return statemodel_x, statemodel_y, relative_x_out, relative_y_out

    def statemodel_arm(self, x):

        relative_x_out, relative_y_out, relative_z_out,relativemod = self.relativemodel_arm(x)

        state_layer1 = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer1")(x)
        state_layer1_do = Dropout(0.5)(state_layer1)
        merge_state_local_layer = concatenate([state_layer1_do, relativemod])
        state_layer2 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer2")(merge_state_local_layer)
        state_layer2_do = Dropout(0.5)(state_layer2)

        merge_state_local_layer = concatenate([state_layer2_do, relativemod])  # layer merges here

        state_layer3 = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="state_layer3")(merge_state_local_layer)
        state_layer3_do = Dropout(0.5)(state_layer3)

        statemodel_x = Dense(10, activation="softmax", name="State_x")(state_layer3_do)  # 0<=x<=8  +1 nobot state
        statemodel_y = Dense(18, activation="softmax", name="State_y")(state_layer3_do)  # 0<=y<=16 +1 nobot state
        statemodel_z = Dense(18, activation="softmax", name="State_z")(state_layer3_do)  # 0<=y<=16 +1 nobot state

        return statemodel_x, statemodel_y, statemodel_z,relative_x_out, relative_y_out, relative_z_out


    def create_graph_n_load(self):
        cnn_stateXY_model, input_stateXY = self.create_cnn_model_time_fn()
        cnn_action_model, input_action = self.create_cnn_model_time_fn_action()
        statemodel_x, statemodel_y, relative_x_out, relative_y_out = self.statemodel(cnn_stateXY_model)
        statemodel_o,XYtheta_model = self.o_model(cnn_action_model, cnn_stateXY_model)
        action_model = self.rnn_part_timefn(cnn_action_model,XYtheta_model)
        self.model = Model(inputs=[input_stateXY, input_action],outputs=[action_model, statemodel_x, statemodel_y, statemodel_o, relative_x_out,relative_y_out])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights(self.model_path)
        self.graph1=tf.get_default_graph()

    def getmodelinfo(self):

        if os.path.exists("snapshot/nowt.h5") == False:
            cnn_stateXY_model, input_stateXY = self.create_cnn_model_time_fn()
            cnn_action_model, input_action = self.create_cnn_model_time_fn_action()
            statemodel_x, statemodel_y, relative_x_out, relative_y_out = self.statemodel(cnn_stateXY_model)
            statemodel_o = self.o_model(cnn_action_model, cnn_stateXY_model)
            action_model = self.rnn_part_timefn(cnn_action_model)
            self.model = Model(inputs=[input_stateXY, input_action],
                               outputs=[action_model, statemodel_x, statemodel_y, statemodel_o, relative_x_out, relative_y_out])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.getNetworkInformation()
            save = input("Do you want to Save the model? y/n")
            if 'y' in str(save):
                self.model.save("snapshot/nowt.h5")
        else:
            print("Model found will continue from stored model")
            self.model = load_model(self.model_path)
            self.getNetworkInformation()

    def save_kfolds(self, train_index, test_index, i):
        '''

        :param train_index: The train data index as an array for a single fold
        :param test_index: The test data index as an array for a single fold
        :param i: The nth kfold
        :return:
        '''

        print("Saving Kfold TRAIN results to Disk")
        np.save(self.kfolds_train_path + "statelables_x_" + str(i), self.statelables_x[train_index])
        np.save(self.kfolds_train_path + "statelables_y_" + str(i), self.statelables_y[train_index])
        np.save(self.kfolds_train_path + "statelables_o_" + str(i), self.statelables_o[train_index])
        np.save(self.kfolds_train_path + "statelables_local_" + str(i), self.statelables_local[train_index])
        np.save(self.kfolds_train_path + "relativeX_" + str(i), self.relativeX[train_index])
        np.save(self.kfolds_train_path + "relativeY_" + str(i), self.relativeY[train_index])
        np.save(self.kfolds_train_path + "actionlables_" + str(i), self.actionlables[train_index])
        np.save(self.kfolds_train_path + "dataset_o_" + str(i), self.dataset_o[train_index])
        np.save(self.kfolds_train_path + "dataset_s_" + str(i), self.dataset_s[train_index])

        print("Saving Kfold TEST results to Disk")
        np.save(self.kfolds_test_path + "statelables_x_" + str(i), self.statelables_x[test_index])
        np.save(self.kfolds_test_path + "statelables_y_" + str(i), self.statelables_y[test_index])
        np.save(self.kfolds_test_path + "statelables_o_" + str(i), self.statelables_o[test_index])
        np.save(self.kfolds_test_path + "statelables_local_" + str(i), self.statelables_local[test_index])
        np.save(self.kfolds_test_path + "relativeX_" + str(i), self.relativeX[test_index])
        np.save(self.kfolds_test_path + "relativeY_" + str(i), self.relativeY[test_index])
        np.save(self.kfolds_test_path + "actionlables_" + str(i), self.actionlables[test_index])
        np.save(self.kfolds_test_path + "dataset_o_" + str(i), self.dataset_o[test_index])
        np.save(self.kfolds_test_path + "dataset_s_" + str(i), self.dataset_s[test_index])

    def load_kfolds(self, i):
        '''

        :param train_index: The train data index as an array for a single fold
        :param test_index: The test data index as an array for a single fold
        :param i: The nth kfold
        :return:
        '''

        print("Loading Kfold TEST results from Disk")
        self.statelables_x = np.load(self.kfolds_test_path + "statelables_x_" + str(i))
        self.statelables_y = np.load(self.kfolds_test_path + "statelables_y_" + str(i))
        self.statelables_o = np.load(self.kfolds_test_path + "statelables_o_" + str(i))
        self.statelables_local = np.load(self.kfolds_test_path + "statelables_local_" + str(i))
        self.relativeX = np.load(self.kfolds_test_path + "relativeX_" + str(i))
        self.relativeY = np.load(self.kfolds_test_path + "relativeY_" + str(i))
        self.dataset_o = np.load(self.kfolds_test_path + "dataset_o_" + str(i))
        self.dataset_s = np.load(self.kfolds_test_path + "dataset_s_" + str(i))

    def traingap_kfold_gen(self):
        '''
        This method starts a training with k kfolds for a single time gap, If you wanted range then use the method traingaprange_kfold
        :param time: The Time gap as an int i.e the time between frames
        :return: History object which contains the history of the training
        '''

        logs_base_path = self.logs_dir
        print("Intitialized values: ")
        print("Time Gap :", str(self.TIME_GAP))
        print("Kfold:", str(self.kfolds))
        print("Batch Size:", str(self.BATCH_SIZE))
        # kfold_history = []
        from preprocess_bot import data_gen
        dg = data_gen(timegap=1)
        for t in tqdm(range(5, 11)):
            dg.TIME_GAP = t
            dg.data_preprocess(file_loc=self.datapath)
        dg.get_numpy_crop()
        dg.gen_labels()
        dg.kfold_shuffle(5)

        k = int(input("Enter k fold number"))
        self.kfolds=k
        print("****** Starting Training for Kfold = ", str(k), " ******")
        if os.path.exists(self.model_path_kfold + str(k) + ".h5") == False:
            cnn_stateXY_model, input_stateXY = self.create_cnn_model_time_fn()
            cnn_action_model, input_action = self.create_cnn_model_time_fn_action()
            statemodel_x, statemodel_y, relative_x_out, relative_y_out = self.statemodel(cnn_stateXY_model)
            statemodel_o,XYtheta_model = self.o_model(cnn_action_model, cnn_stateXY_model)
            action_model = self.rnn_part_timefn(cnn_action_model,XYtheta_model)
            self.model = Model(inputs=[input_stateXY, input_action],
                               outputs=[action_model, statemodel_x, statemodel_y, statemodel_o, relative_x_out, relative_y_out])
            self.getNetworkInformation()
        else:
            print("Model found will continue from stored model")
            if self.model == None:
                self.model = load_model(self.model_path_kfold + str(k) + ".h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_gen = dg.gen_data_train(k, batch_size=self.BATCH_SIZE)
        test_gen = dg.gen_data_test(k, batch_size=self.BATCH_SIZE)
        self.logs_dir = logs_base_path + "k_folds/" + str(k)

        kfold_history=self.model.fit_generator(train_gen, steps_per_epoch=dg.TRAIN_SAMPLES / self.BATCH_SIZE, validation_steps=dg.TEST_SAMPLES / self.BATCH_SIZE,
                                                      validation_data=test_gen, epochs=self.EPOCHS, callbacks=self.callback_create(), workers=2
                                                      , verbose=1)
        self.model.save(self.model_path_kfold + str(k) + ".h5")
        
        msg = "\n***** EVAL Result*******\n"
        score = self.model.evaluate_generator(test_gen, steps=dg.TEST_SAMPLES / self.BATCH_SIZE)
        names = self.model.metrics_names
        for s in range(len(score)):
            msg += str(names[s]) + " : " + str(score[s]) + "\n"
            print(str(names[s]) + " : " + str(score[s]))
        sendmail("Time Gap : " + str(self.TIME_GAP) + "\nKfold " + str(k) + "\nResult :" + str(kfold_history.history) + msg, subject="Kfold")

        self.model = None
        return kfold_history, score

    def traingap_kfold_gen_arm(self):
        '''
        This method starts a training with k kfolds for a single time gap, If you wanted range then use the method traingaprange_kfold
        :param time: The Time gap as an int i.e the time between frames
        :return: History object which contains the history of the training
        '''

        logs_base_path = self.logs_dir
        print("Intitialized values: ")
        print("Time Gap :", str(self.TIME_GAP))
        print("Kfold:", str(self.kfolds))
        print("Batch Size:", str(self.BATCH_SIZE))
        from src.neuralnet.preprocess_arm import data_gen
        dg = data_gen(timegap=1)
        for t in tqdm(range(5, 11)):
            dg.TIME_GAP = t
            dg.data_preprocess(file_loc=self.datapath)
        dg.get_numpy_crop()
        dg.gen_labels()
        dg.kfold_shuffle(5)

        k = int(input("Enter k fold number"))
        self.kfolds=k
        print("****** Starting Training for Kfold = ", str(k), " ******")
        if os.path.exists(self.model_path_kfold + str(k) + ".h5") == False:
            cnn_stateXY_model, input_stateXY = self.create_cnn_model_time_fn()
            cnn_action_model, input_action = self.create_cnn_model_time_fn_action()
            statemodel_x, statemodel_y,statemodel_z, relative_x_out, relative_y_out ,relative_z_out= self.statemodel_arm(cnn_stateXY_model)
            statemodel_o,XYtheta_model = self.o_model(cnn_action_model, cnn_stateXY_model)
            action_model = self.rnn_part_timefn_arm(cnn_action_model,XYtheta_model)
            self.model = Model(inputs=[input_stateXY, input_action],
                               outputs=[action_model, statemodel_x, statemodel_y,statemodel_z, relative_x_out, relative_y_out ,relative_z_out])
            self.getNetworkInformation()
        else:
            print("Model found will continue from stored model")
            if self.model == None:
                self.model = load_model(self.model_path_kfold + str(k) + ".h5")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_gen = dg.gen_data_train(k, batch_size=self.BATCH_SIZE)
        test_gen = dg.gen_data_test(k, batch_size=self.BATCH_SIZE)
        self.logs_dir = logs_base_path + "k_folds/" + str(k)

        kfold_history=self.model.fit_generator(train_gen, steps_per_epoch=dg.TRAIN_SAMPLES / self.BATCH_SIZE, validation_steps=dg.TEST_SAMPLES / self.BATCH_SIZE,
                                                      validation_data=test_gen, epochs=self.EPOCHS, callbacks=self.callback_create(), workers=2
                                                      , verbose=1)
        self.model.save(self.model_path_kfold + str(k) + ".h5")
        # score, names = self.evalute_post_data_import(test_index)
        msg = "\n***** EVAL Result*******\n"

        score = self.model.evaluate_generator(test_gen, steps=dg.TEST_SAMPLES / self.BATCH_SIZE)
        names = self.model.metrics_names
        for s in range(len(score)):
            msg += str(names[s]) + " : " + str(score[s]) + "\n"
            print(str(names[s]) + " : " + str(score[s]))
        sendmail("Time Gap : " + str(self.TIME_GAP) + "\nKfold " + str(k) + "\nResult :" + str(kfold_history.history) + msg, subject="Kfold")

        self.model = None
        return kfold_history, score

    def evaluate(self):
        '''

        :return:
        '''
        model = load_model(self.model_path)
        self.gettestdata()
        print("Evaluating")
        score = model.evaluate([self.testdataset_s, self.testdataset_o],
                               [self.testactionlables, self.teststatelables_x, self.teststatelables_y, self.teststatelables_o, self.testrelativeX,
                                self.testrelativeY],
                               batch_size=40)
        return score, model.metrics_names

    def evaluate_kfold(self):
        for i in range(5):
            model = load_model(self.model_path_kfold + str(i) + ".h5")
            self.load_kfolds(i)
            print("Evaluating")
            score = model.evaluate([self.dataset_s, self.dataset_o],
                                   [self.actionlables, self.statelables_x, self.statelables_y, self.statelables_o, self.relativeX,
                                    self.relativeY],
                                   batch_size=40)
            names = model.metrics_names
            msg = "\n***** EVAL Result*******\n"
            for s in range(len(score)):
                msg += str(names[s]) + " : " + str(score[s]) + "\n"
                print(str(names[s]) + " : " + str(score[s]))
            sendmail("Time Gap : " + str(self.TIME_GAP) + "\nKfold " + str(i) + msg, subject="Kfold")


    def detect(self, frame1rgb, frame1depth, frame2rgb, frame2_depth, frame3rgb, frame3depth, sframergb, sframedepth):
        '''
        :param frame1rgb:
        :param frame1depth:
        :param frame2rgb:
        :param frame2_depth:
        :param frame3rgb:
        :param frame3depth:
        :param sframergb:
        :param sframedepth:
        :return:
        '''
        
        if os.path.exists(self.model_path) == True:
            # Have to convert to 4-D array of images

            # [[t-2],[t-1][t]] has to be in this format
            #burst is the cropped one
            burst = self.imnormalize(np.array(
                [np.concatenate((frame3rgb, frame3depth), axis=2), np.concatenate((frame2rgb, frame2_depth), axis=2),
                 np.concatenate((frame1rgb, frame1depth), axis=2)], dtype=np.float32))
            sframe = self.imnormalize(np.array(np.concatenate((sframergb, sframedepth), axis=2), dtype=np.float32))
            with self.graph1.as_default():
                predictionaction, predictionstateX, predictionstateY, predictionstateO, _, _ = self.model.predict([np.array([sframe]), np.array([burst])])

            predictionstateX = np.argmax(predictionstateX)
            predictionstateY = np.argmax(predictionstateY)
            predictionstateO = np.argmax(predictionstateO)
            predictionaction = np.argmax(predictionaction)

            return predictionaction, predictionstateX, predictionstateY, predictionstateO

        else:
            print("Error !! Cant find trained model at location [ "+ str(self.model_path)+" ] Exiting!!\nPlease Check the Model location and try again")
            exit(0)

    def generatetestdata(self):

        print("******************** Generating Test data **********************")
        self.TIME_GAP = 5
        self.dataSet = self.data_preprocess(self.datapath_test)
        self.get_numpy_crop(test=True)
        self.getlables_val()