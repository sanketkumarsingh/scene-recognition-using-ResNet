from PIL import Image
import cv2
import os
import glob
import numpy as np
import timeit

dir_prefix1 = "/home/chenyang/Desktop/SUN397"
dir_prefix = "/home/chenyang/Desktop/SUN397/"
label_file = dir_prefix + "ClassNameSmall.txt"
out1 = np.array([], np.uint8)
out2 = np.array([], np.uint8)
out3 = np.array([], np.uint8)
count1 = 0
count2 = 0
count3 = 0
lines = 0
error_file = open ( "errorlog.txt" , "wb")
file_num_train = 0
file_num_val = 0
file_num_test = 0
with open(label_file) as f:
    for label_dir in f:
        lines = lines + 1
    #    label_dir = f.readline()
        dir = dir_prefix1 + label_dir
        dir = dir.strip("\n")
        dir = dir + "/"
        print "==============================="
        listing = os.listdir(dir)

        num_of_files = listing.__len__()
        num_files_for_train = int(round(num_of_files * 0.9))
        num_files_for_val = int(round(num_of_files * 0))
        num_files_for_test = int(round(num_of_files * 0.1))
        files_for_train = listing[0:num_files_for_train-1]
        files_for_val = listing[num_files_for_train:num_files_for_train+num_files_for_val-1]
        files_for_test = listing[num_files_for_train+num_files_for_val:num_files_for_train+num_files_for_val+num_files_for_test-1]

        for file_train in files_for_train:
            start = timeit.default_timer()
            print dir + file_train
            image = cv2.imread(dir + file_train)
            if image is None:
                error_file.write(dir + file_train + "\n")
                print "xDDDD\n"
                continue

            resized_image = cv2.resize(image, (224, 224))
            im = Image.fromarray(resized_image)
            im = (np.array(im))
            r = im[:,:,0].flatten()
            g = im[:,:,1].flatten()
            b = im[:,:,2].flatten()
            if lines <= 255:
                zero = [0]
                label = [lines]
            else:
                zero = [lines - 255]
                label = [255]

            imgArray = np.array(list(zero) + list(label) + list(r) + list(g) + list(b),np.uint8)
            out1 = np.ma.concatenate([out1, imgArray ])
            count1 = count1 + 1

            if count1%100 == 0:
                file_num_train = file_num_train + 1
                file_name = "batch/train2/train_" + str(file_num_train) + ".bin"
                newFile = open ( file_name , "wb")
                newFile.write(out1)
                newFile.close()
                out1 = np.array([], np.uint8)


            stop = timeit.default_timer()
            print stop - start

        for file_val in files_for_val:
            start = timeit.default_timer()
            print dir + file_val
            image = cv2.imread(dir + file_val)
            if image is None:
                error_file.write(dir + file_val + "\n")
                print "xDDDD\n"
                continue

            resized_image = cv2.resize(image, (224, 224))
            im = Image.fromarray(resized_image)
            im = (np.array(im))
            r = im[:,:,0].flatten()
            g = im[:,:,1].flatten()
            b = im[:,:,2].flatten()
            if lines <= 255:
                zero = [0]
                label = [lines]
            else:
                zero = [lines - 255]
                label = [255]

            imgArray = np.array(list(zero) + list(label) + list(r) + list(g) + list(b),np.uint8)
            out2 = np.ma.concatenate([out2, imgArray ])
            count2 = count2 + 1

            if count2%100 == 0:
                file_num_val = file_num_val + 1
                file_name = "batch/val2/val_" + str(file_num_val) + ".bin"
                newFile = open ( file_name , "wb")
                newFile.write(out2)
                newFile.close()
                out2 = np.array([], np.uint8)
            stop = timeit.default_timer()
            print stop - start

        for file_test in files_for_test:
            start = timeit.default_timer()
            print dir + file_test
            image = cv2.imread(dir + file_test)
            if image is None:
                error_file.write(dir + file_test + "\n")
                print "xDDDD\n"
                continue

            resized_image = cv2.resize(image, (224, 224))
            im = Image.fromarray(resized_image)
            im = (np.array(im))
            r = im[:,:,0].flatten()
            g = im[:,:,1].flatten()
            b = im[:,:,2].flatten()
            if lines <= 255:
                zero = [0]
                label = [lines]
            else:
                zero = [lines - 255]
                label = [255]

            imgArray = np.array(list(zero) + list(label) + list(r) + list(g) + list(b),np.uint8)
            out3 = np.ma.concatenate([out3, imgArray ])
            count3 = count3 + 1

            if count3%100 == 0:
                file_num_test = file_num_test + 1
                file_name = "batch/test2/test_" + str(file_num_test) + ".bin"
                newFile = open ( file_name , "wb")
                newFile.write(out3)
                newFile.close()
                out3 = np.array([], np.uint8)
            stop = timeit.default_timer()
            print stop - start




file_num_train = file_num_train + 1
file_name = "batch/train/train_" + str(file_num_train) + ".bin"
newFile = open ( file_name , "wb")
newFile.write(out1)
newFile.close()


file_num_val = file_num_val + 1
file_name = "batch/val/val_" + str(file_num_val) + ".bin"
newFile = open ( file_name , "wb")
newFile.write(out2)
newFile.close()

file_num_test = file_num_test + 1
file_name = "batch/test/test_" + str(file_num_test) + ".bin"
newFile = open ( file_name , "wb")
newFile.write(out3)
newFile.close()




#image = cv2.imread("test.jpg")
#resized_image = cv2.resize(image, (100, 100))

#im = Image.fromarray(resized_image)
#im = (np.array(im))
#out = np.array([], np.uint8)



