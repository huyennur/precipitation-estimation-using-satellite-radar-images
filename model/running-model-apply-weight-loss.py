import os
# from osgeo import gdal
import numpy as np
import cv2
import math
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import random
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator


with tf.device('/device:GPU:1'):

    data_train = pd.read_csv(
        "/home/thanh/huyendn/csv/BTB_512/dataset2_BTB_train.csv")
    data_val = pd.read_csv(
        "/home/thanh/huyendn/csv/BTB_512/dataset2_BTB_test.csv")


    data_train.drop(data_train.filter(regex="Unnamed"), axis=1, inplace=True)
    data_val.drop(data_val.filter(regex="Unnamed"), axis=1, inplace=True)

    train_path = data_train['0'].to_list()
    val_path = data_val['0'].to_list()
    
    print(len(train_path), len(val_path))

    input_image_path = "/home/thanh/huyendn/dataset2/himawari"
    label_image_path = "/home/thanh/huyendn/dataset2/radar"

    input_image_path_isor = "/home/thanh/huyendn/dataset2/ISOR"
    input_image_path_cape = "/home/thanh/huyendn/dataset2/CAPE"
    input_image_path_tcc = "/home/thanh/huyendn/dataset2/TCC"
    input_image_path_tcw = "/home/thanh/huyendn/dataset2/TCW"
    input_image_path_tcwv = "/home/thanh/huyendn/dataset2/TCWV"
    input_dem = "/home/thanh/huyendn/dataset2/DEM_origin1.tif"


    class loadDataset(tf.keras.utils.Sequence):

        def __init__(self, paths, batch_size, shuffle=True):
            self.paths = paths
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

            self.image_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

        def on_epoch_end(self):
            random.shuffle(self.paths)

        def __len__(self):
            return math.ceil(len(self.paths) / self.batch_size)

        def __getitem__(self, idx):
            batch_path = self.paths[idx *
                                    self.batch_size:(idx + 1) * self.batch_size]

            imagess = []
            labelss = []

            for file_name in range(len(batch_path)):
                path_image = input_image_path + "/" + batch_path[file_name]
                path_label = label_image_path + "/" + batch_path[file_name]

                input_path_isor = os.path.join(input_image_path_isor, batch_path[file_name])
                input_path_cape = os.path.join(input_image_path_cape, batch_path[file_name])
                input_path_tcc = os.path.join(input_image_path_tcc, batch_path[file_name])
                input_path_tcw = os.path.join(input_image_path_tcw, batch_path[file_name])
                input_path_tcwv = os.path.join(input_image_path_tcwv, batch_path[file_name])
        
                images = []
                labels = []

                list_band = []
     
                for i in sorted(os.listdir(path_image)):
                    if(("B09B" in i) or ("B10B" in i) or ("B11B" in i) or ("B16" in i) or ("I2B" in i) or ("IRB" in i) or ("WVB" in i)):
                        i_image_path = os.path.join(path_image, i)
                        list_band.append(i_image_path)
                list_band.sort()
                i_b09 = cv2.imread(list_band[0], cv2.IMREAD_UNCHANGED)
                i_b10 = cv2.imread(list_band[1], cv2.IMREAD_UNCHANGED)
                i_b11 = cv2.imread(list_band[2], cv2.IMREAD_UNCHANGED)
                i_b16 = cv2.imread(list_band[3], cv2.IMREAD_UNCHANGED)
                i_i2b = cv2.imread(list_band[4], cv2.IMREAD_UNCHANGED)
                i_irb = cv2.imread(list_band[5], cv2.IMREAD_UNCHANGED)
                i_wvb = cv2.imread(list_band[6], cv2.IMREAD_UNCHANGED)

                # image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                diff1 = np.subtract(i_b10, i_b16)
                diff2 = np.subtract(i_b11, i_irb)
                diff3 = np.subtract(i_irb, i_i2b)
                diff4 = np.subtract(i_wvb, i_b09)
                diff5 = np.subtract(i_b09, i_b10)

                # image = image/350.0
                i_irb = (i_irb - 0)/305.19412
                diff1 = (diff1 - (-68.58449))/328.83559
                diff2 = (diff2 - (-144.07571))/438.47256
                diff3 = (diff3 - (-53.921143))/133.800843
                diff4 = (diff4 - (-126.18439))/245.08643
                diff5 = (diff5 - (-260.2511))/340.5816

                images.append(i_irb)
                images.append(diff1)
                images.append(diff2)
                images.append(diff3)
                images.append(diff4)
                images.append(diff5)

                for filename_isor in sorted(os.listdir(input_path_isor)):
                    i_image_path = os.path.join(input_path_isor, filename_isor)
                    image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                    image = image/0.8358923
                    images.append(image)
            
                for filename_cape in sorted(os.listdir(input_path_cape)):
                    i_image_path = os.path.join(input_path_cape, filename_cape)
                    image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                    image = image/5446.3584
                    images.append(image)

                for filename_tcc in sorted(os.listdir(input_path_tcc)):
                    i_image_path = os.path.join(input_path_tcc, filename_tcc)
                    image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                    image = image/1.0000076
                    images.append(image)
                
                for filename_tcw in sorted(os.listdir(input_path_tcw)):
                    i_image_path = os.path.join(input_path_tcw, filename_tcw)
                    image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                    image = (image - 10.41)/84.22
                    images.append(image)
                
                for filename_tcwv in sorted(os.listdir(input_path_tcwv)):
                    i_image_path = os.path.join(input_path_tcwv, filename_tcwv)
                    image = cv2.imread(i_image_path, cv2.IMREAD_UNCHANGED)
                    image = (image - 10.41)/69.94
                    images.append(image)

                dem_img = cv2.imread(input_dem, cv2.IMREAD_UNCHANGED)
                dem_img = dem_img/32767.0
                images.append(dem_img)

                for l in os.listdir(path_label):
                    i_lab_path = os.path.join(path_label, l)
                    label = cv2.imread(i_lab_path, cv2.IMREAD_UNCHANGED)
                    labels.append(label)

                imagess.append(images)
                labelss.append(labels)

            image_tensor = tf.convert_to_tensor(
                tf.constant(imagess), dtype=tf.float32)
            label_tensor = tf.convert_to_tensor(
                tf.constant(labelss), dtype=tf.float32)

            image_tensor = np.moveaxis(image_tensor, 1, 3)
            label_tensor = np.moveaxis(label_tensor, 1, 3)

            return [image_tensor, label_tensor]

    def unet2(input_shape):
        inputs = Input(shape=input_shape, name="input")
        
        # 512
        down0a = Conv2D(16, (3, 3), padding='same')(inputs)
        down0a = BatchNormalization()(down0a)
        down0a = Activation('relu')(down0a)
        down0a = Conv2D(16, (3, 3), padding='same')(down0a)
        down0a = BatchNormalization()(down0a)
        down0a = Activation('relu')(down0a)
        down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
        # 256
        down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0 = Conv2D(32, (3, 3), padding='same')(down0)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
        # 128
        down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(64, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
        # 64
        down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(128, (3, 3), padding='same')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
        # 8
        center = Conv2D(1024, (3, 3), padding='same')(down2_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(1024, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        # center
        up2 = UpSampling2D((2, 2))(center)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        # 64
        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        # 128
        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(32, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(32, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(32, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        # 256
        up0a = UpSampling2D((2, 2))(up0)
        up0a = concatenate([down0a, up0a], axis=3)
        up0a = Conv2D(16, (3, 3), padding='same')(up0a)
        up0a = BatchNormalization()(up0a)
        up0a = Activation('relu')(up0a)
        up0a = Conv2D(16, (3, 3), padding='same')(up0a)
        up0a = BatchNormalization()(up0a)
        up0a = Activation('relu')(up0a)
        up0a = Conv2D(16, (3, 3), padding='same')(up0a)
        up0a = BatchNormalization()(up0a)
        up0a = Activation('relu')(up0a)
        # 512
        output = Conv2D(1, (1, 1), activation='relu')(up0a)
        
        model = Model(inputs=inputs, outputs=output)
        return model

    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()


    input_shape = (512, 512, 12) # Update input shape based on your data

    # simple_model_unet = Unet(n_filters = 16, dropout = 0.1, batchnorm = True)
    simple_model_unet = unet2(input_shape)

    simple_model_unet.summary()

    epochs = 500
    batch_size = 16
    train_data = loadDataset(train_path, batch_size, shuffle=True)
    val_data = loadDataset(val_path, batch_size, shuffle=True)

    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()


    train_loss_tracker = tf.keras.metrics.Mean(name="loss")
    train_mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
    val_loss_tracker = tf.keras.metrics.Mean(name="loss")
    val_mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)


    @tf.function
    def train_step(x, y, weight_map, model):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            mask = tf.not_equal(y, -99999)
            # y = tf.boolean_mask(y, mask)
            # y_pred = tf.boolean_mask(y_pred, mask)
            loss = weighted_mse_loss(y[mask], y_pred[mask], weight_map)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss_tracker.update_state(loss)
        train_mse_metric.update_state(y[mask], y_pred[mask])

        return loss


    @tf.function
    def val_step(x, y, weight_map, model):

        y_pred = model(x, training=False)
        mask = tf.not_equal(y, -99999)
        # y = tf.boolean_mask(y, mask)
        # y_pred = tf.boolean_mask(y_pred, mask)
        loss = weighted_mse_loss(y[mask], y_pred[mask], weight_map)

        val_loss_tracker.update_state(loss)
        val_mse_metric.update_state(y[mask], y_pred[mask])

        return loss

    def weighted_mse_loss(y_true, y_pred, weight_map):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        weighted_mse = tf.multiply(mse, weight_map)
        return tf.reduce_mean(weighted_mse, axis=-1)

    train_loss = []
    val_losses = []
    train_mse = []
    val_mse = []


    def running():
        
        for epoch in range(epochs):

            train_loss_tracker.reset_states()
            train_mse_metric.reset_states()
            val_loss_tracker.reset_states()
            val_mse_metric.reset_states()

            # print("train_data", len(train_data))
            print("Epoch {}/{}".format(epoch + 1, epochs))

            for i, (x, y) in enumerate(train_data):
                max_value = np.max(y)
                y_norm = y / max_value
                scaling_factor = 0.01
                threshold = 0.2
                weight_map = 1.0 / (1.0 + np.exp(-((y_norm - threshold) * scaling_factor)))
                
                loss = train_step(x, y, weight_map, simple_model_unet)
                # print("Training loss (for one batch) at step %d: %.4f" %
                #     (i, float(loss)))

            print('---- Training ----', end=" ")
            print('Loss  =  %.4f' % (train_loss_tracker.result().numpy()), end=" ")
            print('Metric mse  =  %.4f' % (train_mse_metric.result().numpy()), end=" ")

            train_loss.append(train_loss_tracker.result().numpy())
            train_mse.append(train_mse_metric.result().numpy())

            # print("test_data", len(val_data))
            # print("Epoch {}/{}".format(epoch + 1, epochs))

            for j, (x, y) in enumerate(val_data):
                max_value = np.max(y)
                y_norm = y / max_value
                scaling_factor = 0.01
                threshold = 0.2
                weight_map = 1.0 / (1.0 + np.exp(-((y_norm - threshold) * scaling_factor)))

                val_loss = val_step(x, y, weight_map, simple_model_unet)

                # print("Val loss (for one batch) at step %d: %.4f" %
                #     (j+1, float(val_loss)))


            print('--- Validation ---', end=" ")
            print('Loss  =  %.4f' % (val_loss_tracker.result().numpy()) , end=" ")
            print('Metric mse  =  %.4f' % (val_mse_metric.result().numpy()))

            val_losses.append(val_loss_tracker.result().numpy())
            val_mse.append(val_mse_metric.result().numpy())

            train_data.on_epoch_end()
            val_data.on_epoch_end()

    running()

    simple_model_unet.save('/home/thanh/huyendn/stage4/models/model18_3.h5')

    # transpose lists to create rows
    rows = zip(train_loss, train_mse, val_losses, val_mse)

    # save rows to CSV file
    with open('/home/thanh/huyendn/stage4/history/history18_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_loss', 'train_metric_mse', 'val_loss', 'val_metric_mse'])
        writer.writerows(rows)


