
import tensorflow as tf
import numpy as np
import os
import cv2
import time
import glob
import dataset


try:

    # Path of  training images
    train_path = r'D:\classification\data\train'
    if not os.path.exists(train_path):
        print("No such directory")
        raise Exception
    # Path of testing images
    dir_path = r'D:\classification\data\test'
    if not os.path.exists(dir_path):
        print("No such directory")
        raise Exception
    classes = os.listdir(dir_path)

    image_size = 64
    num_channels = 3
    batch_size = 1

    #test image 읽기
    data = dataset.read_test_sets(dir_path, image_size, classes)
    num_image = data.test.num_examples
    #images, labels, img_names, cls =
    print("Number of files in Test-set:\t{}".format(num_image))


    # Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('models/trained_model-24000.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    total = 0
    correct = 0
    
    start = time.time()

    for i in range(num_image):
        x_batch, y_true_batch, _, cls_batch = data.test.next_batch(batch_size)

        y_pred = graph.get_tensor_by_name("y_pred:0")
        ## Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, len(os.listdir(dir_path))))

        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]

        # Convert np.array to list
        a = result[0].tolist()
        r = 0

        # Finding the maximum of all outputs
        max1 = max(a)
        index1 = a.index(max1)
        predicted_class = None

        # Walk through directory to find the label of the predicted output
        count = 0
        for root, dirs, files in os.walk(train_path):
            for name in dirs:
                if count == index1:
                    predicted_class = name
                count += 1


        # If the maximum confidence output is largest of all by a big margin then
        # print the class or else print a warning
        for i in a:
            if i != max1:
                if max1 - i < i:
                    r = 1

        total = total + 1

        if cls_batch == predicted_class:
            correct = correct + 1
        else:
            print("Predcition ")
            print(cls_batch)
            print(predicted_class)

    # Calculate execution time
    end = time.time()
    dur = end - start
    print("")

    print("Accuarcy : %.2f %%" % (correct*100/total))

except Exception as e:
    print("Exception:", e)


if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")