import dl
import os,codecs,numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
# encode text category labels
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os.path
from random import shuffle
from math import ceil
import h5py




datapath = '../Malaria_Dataset/'

SUBTRACT_MEAN = False



def read_images():
    base_dir = os.path.join('../cell_images')
    infected_dir = os.path.join(base_dir,'Parasitized')
    healthy_dir = os.path.join(base_dir,'Uninfected')

    infected_files = glob.glob(infected_dir+'/*.png')
    healthy_files = glob.glob(healthy_dir+'/*.png')

    np.random.seed(42)

    files_df = pd.DataFrame({
        'filename': infected_files + healthy_files,
        'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    files_df.head()
    return files_df

def train_test_set(files_df):
    train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                          files_df['label'].values,
                                                                          test_size=0.3,
                                                                          random_state=42)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                        train_labels,
                                                                        test_size=0.1, random_state=42)

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


path_train = "../Malaria_Dataset/cell_images_28/train/"
path_val = "../Malaria_Dataset/cell_images_28/validation/"
path_test = "../Malaria_Dataset/cell_images_28/test/"


def load_images(train_files, train_labels, val_files,  test_files, mean, hdf5_file, IMG_DIMS):
    num = 0
    # loop over train addresses
    for i in range(len(train_files)):
        # print how many images are saved every 1000 images
        if i % 5000 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_files)))
        addr = train_files[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, dsize=IMG_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        img = np.array(img, dtype=np.float32)

        if train_labels[i] == "malaria":
            cv2.imwrite(path_train + str(1) + "/" +str(num)+".png", img)
        else:
            cv2.imwrite(path_train+ str(0) + "/" +str(num)+".png", img)
        num += 1
        hdf5_file["train_img"][i, ...] = img
        mean += img / float(len(train_labels))

    # loop over validation addresses
    for i in range(len(val_files)):
        # print how many images are saved every 1000 images
        if i % 5000 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, len(val_files)))
        addr = val_files[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, dsize=IMG_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        img = np.array(img, dtype=np.float32)

        if train_labels[i] == "malaria":
            cv2.imwrite(path_val + str(1) + "/" +str(num)+".png", img)
        else:
            cv2.imwrite(path_val+ str(0) + "/" +str(num)+".png", img)
        num += 1
        hdf5_file["val_img"][i, ...] = img

    # loop over test addresses
    for i in range(len(test_files)):
        # print how many images are saved every 1000 images
        if i % 5000 == 0 and i > 1:
            print('Test data: {}/{}'.format(i, len(test_files)))
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = test_files[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, dsize=IMG_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        img = np.array(img, dtype=np.float32)
        hdf5_file["test_img"][i, ...] = img


def discover_dataset(train_files):
    shapes = []
    for file_path in train_files:
        shapes.append(cv2.imread(file_path).shape)
    return list(shapes)


if __name__ == '__main__':
    '''
    # in the cell_images.zip file, we have 27558 cell images. Half of them are parasitized and the other half are
    # the images of uninfected cells.
    # Firstly, randomly mix the data one file. Below function will mix all the images in infected and healthy image files
    # into one file.
    files_df = read_images()

    # Now, let's create train, validation, and the test splits of data.
    # %30 of all data will be the test set.
    # The rest %70 of data will be divided into %90 train and %10 validation data.
    train_files, train_labels, val_files, val_labels, test_files, test_labels = train_test_set(files_df)
    print("Train data shape : ", train_files.shape)
    print("Validation data shape : ", val_files.shape)
    print("Test data shape : ", test_files.shape)
    print('Train:', Counter(train_labels), '\nVal:', Counter(val_labels), '\nTest:', Counter(test_labels))


    # After making train, validation and test splits, we need to resize the images, since all of them have different sizes.
    # To find the best size, we will go through all the train set and will find maximum, minimum, median and average
    # image sizes.


    train_img_dims = discover_dataset(train_files)
    print('Min Dimensions:', np.min(train_img_dims, axis=0))
    print('Avg Dimensions:', np.mean(train_img_dims, axis=0))
    print('Median Dimensions:', np.median(train_img_dims, axis=0))
    print('Max Dimensions:', np.max(train_img_dims, axis=0))'''


    IMG_DIM = (28, 28) # Here you can change the size.
    hdf5_file = None
    hdf5_datapath = '../Malaria_Dataset/data_64.hdf5' # Here you can directly define the path to your hdf5 data file.

    if not os.path.isfile(hdf5_datapath):
        train_shape = (len(train_files), 28, 28 , 3)
        val_shape = (len(val_files), 28, 28, 3)
        test_shape = (len(test_files), 28, 28, 3)

        # In the label files, all images anotated either as healthy or malaria.
        # By using the label encoder, we will encode all of them as 0-healthy and 1-malaria
        le = LabelEncoder()
        le.fit(train_labels)
        train_labels_enc = le.transform(train_labels)
        val_labels_enc = le.transform(val_labels)
        test_labels_enc = le.transform(test_labels)
        print(train_labels[:6], train_labels_enc[:6])

        hdf5_file = h5py.File(hdf5_datapath, mode='w')

        hdf5_file.create_dataset("train_img", train_shape, np.float32)
        hdf5_file.create_dataset("val_img", val_shape, np.float32)
        hdf5_file.create_dataset("test_img", test_shape, np.float32)
        hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

        hdf5_file.create_dataset("train_labels", (len(train_labels),), np.int8)
        hdf5_file["train_labels"][...] = train_labels_enc
        hdf5_file.create_dataset("val_labels", (len(val_labels),), np.int8)
        hdf5_file["val_labels"][...] = val_labels_enc
        hdf5_file.create_dataset("test_labels", (len(test_labels),), np.int8)
        hdf5_file["test_labels"][...] = test_labels_enc

        mean = np.zeros(train_shape[1:], np.float32)

        # Here all the images are loaded into hdf5 file.
        load_images(train_files, train_labels, val_files, test_files, mean, hdf5_file, IMG_DIM)
        hdf5_file.close() # After loading of all images hdf5 file is closed.


    # Let's plot some random samples from our training samples, to see our data format properly.

    # The hdf5 file is opened in the read format.
    hdf5_file = h5py.File(hdf5_datapath, "r")

    # the keys to reach the correct data in hdf5 file.
    # hdf5_file["train_img"] --> training images
    # hdf5_file["train_labels"] --> training labels
    # hdf5_file["val_img"] --> validation images
    # hdf5_file["val_labels"] --> validation labels
    # hdf5_file["test_img"] --> test images
    # hdf5_file["test_labels"] --> test labels
    # hdf5_file["train_mean"] --> the mean of all training data.

    plt.figure(1 , figsize = (8 , 8))
    n = 0
    for i in range(16):
        n += 1
        r = np.random.randint(0 , hdf5_file["train_img"].shape[0] , 1)
        plt.subplot(4 , 4 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        plt.imshow(hdf5_file["train_img"][r[0]]/255.)
        plt.title('{}'.format(train_labels[r[0]]))
        plt.xticks([]) , plt.yticks([])
    plt.show()

    # the keys to reach the correct data in hdf5 file.
    # hdf5_file["train_img"] --> training images
    # hdf5_file["train_labels"] --> training labels
    # hdf5_file["val_img"] --> validation images
    # hdf5_file["val_labels"] --> validation labels
    # hdf5_file["test_img"] --> test images
    # hdf5_file["test_labels"] --> test labels
    # hdf5_file["train_mean"] --> the mean of all training data.



    BATCH_SIZE = 10 #You can change the batch size to something else.
    EPOCH_SIZE = 5 # You can change the number of epochs.
    data_num = hdf5_file["train_img"].shape[0] # Total number of samples
    validation_set_num = hdf5_file["val_img"].shape[0]

    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / BATCH_SIZE))))
    shuffle(batches_list)

    # create list of validation batches to shuffle the data
    val_batches_list = list(range(int(ceil(float(validation_set_num) / BATCH_SIZE))))
    shuffle(val_batches_list)


    net =  dl.Network()


    # LENET input image 28*28*3
    
    '''output_size_conv1, num_of_outputs_conv1 = net.conv(hdf5_file['train_img'][0].shape[0],
                                                       hdf5_file['train_img'][0].shape[2],
                                                       5, # filter size
                                                       3, # filter depth
                                                       6, # number of filters
                                                       1, # stride
                                                       0) # padding

    output_size_relu1_h, output_size_relu1_w, num_of_outputs_relu1 = net.relu(
        num_of_outputs_conv1, # number of the feature maps
        output_size_conv1, #height of one of the feature map
        output_size_conv1 # width of one of the feature map
    )

    output_size_pooling1, num_of_outputs_pooling_1 = net.maxpool(
        output_size_relu1_h,
        num_of_outputs_relu1,
        2, # filter size
        2, # stride
        0 # padding
    )

    output_size_conv2, num_of_outputs_conv2 = net.conv(
        output_size_pooling1,
        num_of_outputs_pooling_1,
        5, # filter size
        6, # filter dept
        16, # number of filters
        1, #stride
        0 # padding
    )

    output_size_relu2_h, output_size_relu2_w, num_of_outputs_relu2 = net.relu(
        num_of_outputs_conv2, # number of the feature maps
        output_size_conv2, #height of one of the feature map
        output_size_conv2 # width of one of the feature map
    )

    output_size_pooling2, num_of_outputs_pooling_2 = net.maxpool(
        output_size_relu2_h,
        num_of_outputs_relu2,
        2, # filter size
        2, # stride
        0 # padding
    )

    output_size_fully1_h, output_size_fully1_w, num_of_outputs_fully1 = net.fullyConnected(
        output_size_pooling2,
        output_size_pooling2,
        num_of_outputs_pooling_2,
        32
    )

    output_size_relu4_h, output_size_relu4_w, num_of_outputs_relu4 = net.relu(
        num_of_outputs_fully1, # number of the feature maps
        output_size_fully1_h, # height of one of the feature map
        output_size_fully1_w # width of the feature map
    )

    output_size_fully2_h, output_size_fully2_w, num_of_outputs_fully2 = net.fullyConnected(
        output_size_relu4_h,
        output_size_relu4_w, # 1
        num_of_outputs_relu4,
        2
    )

    output_size_softmax5_h, output_size_softmax5_w, num_of_outputs_sigmoid5 = net.softmax(
        num_of_outputs_fully2, # number of the feature maps
        output_size_fully2_h, #height of one of the feature map,
        output_size_fully2_w # 1
    )'''



    # LENET with receptive field.

    output_size_conv1, num_of_outputs_conv1 = net.conv(hdf5_file['train_img'][0].shape[0],
                                                       hdf5_file['train_img'][0].shape[2],
                                                       3, # filter size
                                                       3, # filter depth
                                                       8, # number of filters
                                                       1, # stride
                                                       0) # padding


    output_size_relu1_h, output_size_relu1_w, num_of_outputs_relu1 = net.relu(
        num_of_outputs_conv1, # number of the feature maps
        output_size_conv1, #height of one of the feature map
        output_size_conv1 # width of one of the feature map
    )

    output_size_conv2, num_of_outputs_conv2 = net.conv(
        output_size_relu1_h,
        num_of_outputs_relu1,
        3, # filter size
        8, # filter dept
        8, # number of filters
        1, #stride
        0 # padding
    )


    output_size_relu2_h, output_size_relu2_w, num_of_outputs_relu2 = net.relu(
        num_of_outputs_conv2, # number of the feature maps
        output_size_conv2, #height of one of the feature map
        output_size_conv2 # width of one of the feature map
    )


    output_size_pooling2, num_of_outputs_pooling_2 = net.maxpool(
        output_size_relu2_h,
        num_of_outputs_relu2,
        2, # filter size
        2, # stride
        0 # padding
    )


    output_size_conv3, num_of_outputs_conv3 = net.conv(
        output_size_pooling2,
        num_of_outputs_pooling_2,
        3, # filter size
        8, # filter dept
        16, # number of filters
        1, #stride
        0 # padding
    )

    output_size_relu3_h, output_size_relu3_w, num_of_outputs_relu3 = net.relu(
        num_of_outputs_conv3, # number of the feature maps
        output_size_conv3, #height of one of the feature map
        output_size_conv3 # width of one of the feature map
    )


    output_size_conv4, num_of_outputs_conv4 = net.conv(
        output_size_relu3_h,
        num_of_outputs_relu3,
        3, # filter size
        16, # filter dept
        16, # number of filters
        1, #stride
        0 # padding
    )


    output_size_relu4_h, output_size_relu4_w, num_of_outputs_relu4 = net.relu(
        num_of_outputs_conv4, # number of the feature maps
        output_size_conv4, #height of one of the feature map
        output_size_conv4 # width of one of the feature map
    )


    output_size_pooling4, num_of_outputs_pooling_4 = net.maxpool(
        output_size_relu4_h,
        num_of_outputs_relu4,
        2, # filter size
        2, # stride
        0 # padding
    )


    output_size_fully1_h, output_size_fully1_w, num_of_outputs_fully1 = net.fullyConnected(
        output_size_pooling4,
        output_size_pooling4,
        num_of_outputs_pooling_4,
        32
    )

    output_size_relu5_h, output_size_relu5_w, num_of_outputs_relu5 = net.relu(
        num_of_outputs_fully1, # number of the feature maps
        output_size_fully1_h, #height of one of the feature map,
        output_size_fully1_w
    )

    output_size_fully2_h, output_size_fully2_w, num_of_outputs_fully2 = net.fullyConnected(
        output_size_relu5_h,
        output_size_relu5_w,
        num_of_outputs_relu5,
        2
    )

    output_size_sigmoid5_h, output_size_sigmoid5_w,  num_of_outputs_sigmoid5 = net.softmax(
        num_of_outputs_fully2, # number of the feature maps
        output_size_fully2_h, #height of one of the feature map,
        output_size_fully2_w
    )



    #net.loadWeights("../Malaria_Dataset/training_sgd_normal_28/training_sgd_28.txt")

    n_values = 2
    train_cost = []
    train_cost10 = []
    iterations = []
    iterations10 = []
    train_acc = 0.0
    val_acc = 0.0

    epoch_axis = []
    train_acc_epoch = []
    val_acc_epoch = []

    images = None

    iter = 0
    iter10 = 0


    # loop over batches
    for epoch in range(EPOCH_SIZE):
        batch_10_cost = 0.0
        for n, i in enumerate(batches_list):
        #for n, i in enumerate(np.random.choice(batches_list, size=len(batches_list),replace=False)):
            i_s = i * BATCH_SIZE  # index of the first image in this batch
            i_e = min([(i + 1) * BATCH_SIZE, data_num])  # index of the last image in this batch
            # read batch images and remove training mean
            images = hdf5_file["train_img"][i_s:i_e, ...]
            images = images.reshape(images.shape[0], images.shape[3],
                                    images.shape[1], images.shape[2]) # reshaped to size 22046, 3, 64, 64

            # read labels and convert to one hot encoding
            labels = hdf5_file["train_labels"][i_s:i_e]
            labels_one_hot = np.eye(n_values)[labels]

            #batch_cost, _ = net.train(images / 255., labels_one_hot, "leastSquaresError",
            #                          1, len(images), 0.005, 0.5)

            batch_cost, _ = net.train(images / 255., labels_one_hot, "crossEntropyError",
                                      1, len(images), 1e-3, 0.5)
            batch_10_cost += batch_cost
            train_cost.append(batch_cost)
            iterations.append(iter)
            iter += 1

            if n % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1,
                                                                   n+1,
                                                                   batch_10_cost / (10)))
                train_cost10.append(batch_10_cost)
                iterations10.append(iter10)
                iter10 += 1
                batch_10_cost = 0.0


        for n, i in enumerate(batches_list):
            i_s = i * BATCH_SIZE  # index of the first image in this batch
            i_e = min([(i + 1) * BATCH_SIZE, data_num])  # index of the last image in this batch
            # read batch images and remove training mean
            images = hdf5_file["train_img"][i_s:i_e, ...]
            images = images.reshape(images.shape[0], images.shape[3],
                                    images.shape[1], images.shape[2]) # reshaped to size 22046, 3, 64, 64

            # read labels and convert to one hot encoding
            labels = hdf5_file["train_labels"][i_s:i_e]
            labels_one_hot = np.eye(n_values)[labels]

            batch_accuracy = net.validation(images / 255., labels_one_hot, BATCH_SIZE)

            train_acc += batch_accuracy
            #print(n+1, '/', len(batches_list))
        train_acc_epoch.append(train_acc / len(batches_list))
        print(epoch + 1, '/', EPOCH_SIZE, '  Train Acc: ', train_acc / len(batches_list))
        train_acc = 0.0

        for n, i in enumerate(val_batches_list):
            i_s = i * BATCH_SIZE  # index of the first image in this batch
            i_e = min([(i + 1) * BATCH_SIZE, validation_set_num])  # index of the last image in this batch
            # read batch images and remove training mean
            images = hdf5_file["val_img"][i_s:i_e, ...]
            images = images.reshape(images.shape[0], images.shape[3],
                                    images.shape[1], images.shape[2]) # reshaped to size 22046, 3, 64, 64

            # read labels and convert to one hot encoding
            labels = hdf5_file["val_labels"][i_s:i_e]
            labels_one_hot = np.eye(n_values)[labels]

            batch_accuracy = net.validation(images / 255., labels_one_hot, BATCH_SIZE)

            val_acc += batch_accuracy
        val_acc_epoch.append(val_acc / len(val_batches_list))
        print(epoch + 1, '/', EPOCH_SIZE, '  Val Acc: ', val_acc / len(val_batches_list))
        val_acc = 0.0
        epoch_axis.append(epoch)
        net.saveWeights("../Malaria_Dataset/training_sgd_receptive_64_epoch_"+ str(epoch+1) + ".txt")


    net.saveWeights("../Malaria_Dataset/training_sgd_receptive_64" + ".txt")

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.plot(iterations, train_cost)
    plt.xlabel('Iteration')
    plt.ylabel('Loss per iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training loss for 10 batches')
    plt.plot(iterations10, train_cost10)
    plt.xlabel('Each 10 batch')
    plt.ylabel('Loss per each 10 batch')

    plt.subplot(3, 1, 3)
    plt.title('Train vs Validation Accuracy')
    plt.plot(epoch_axis, train_acc_epoch, '-o', label='train')
    plt.plot(epoch_axis, val_acc_epoch, '-o', label='val')
    plt.plot([0.5] * len(epoch_axis), 'k--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()









