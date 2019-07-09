import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import argparse
ap= argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
        help="image")
ap.add_argument("-l", "--label_map", required=True,
        help="image")
args = vars(ap.parse_args())
with tf.gfile.FastGFile(args["model"], 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    label_map = label_map_util.load_labelmap(args["label_map"])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=3, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Read and preprocess an image.
    #img = cv.imread('example2.jpg')
   ##rows = img.shape[0]
    #cols = img.shape[1]
    #inp = cv.resize(img, (300, 300))
    #inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        numpy_frame = np.asarray(frame)
        numpy_final = np.expand_dims(numpy_frame, axis=0)

        rows = frame.shape[0]
        cols = frame.shape[1]
    # Run the model

      #  boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
    #  scores = sess.graph.get_tensor_by_name('detection_scores:0')
      #  classes = sess.graph.get_tensor_by_name('detection_classes:0')
      #  num_detections = sess.graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
       # (boxes, scores, classes, num_detections) = sess.run(
        #  [boxes, scores, classes, num_detections],
        #  feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[2], inp.shape[3], 3)})
      # Visualization of the results of a detection.

        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': numpy_final})

    # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        g=0
        for i in range(num_detections):
            classId = int(out[3][0][i])
            label = category_index[classId]
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.2:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                # vis_util.visualize_boxes_and_labels_on_image_array(frame, boxes, classes, scores, category_index, keypoints=None)
                cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                text = "Label: {}".format(label)
                cv.putText(frame, text, (5, 25),  cv.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
        cv.imshow('Frame', frame)
        key = cv.waitKey(1) & 0xFF
                #    cv.waitKey()
        if key == ord("q"):
            break

	# update the FPS counter
fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv.destroyAllWindows()
vs.stop()

