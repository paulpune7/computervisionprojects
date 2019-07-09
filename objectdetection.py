import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS

with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

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
	inp = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
        inp.shape
    # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[2], inp.shape[3], 3)})

    # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        g=0
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.2:
               g += 1
               x = bbox[1] * cols
               y = bbox[0] * rows
               right = bbox[3] * cols
               bottom = bbox[2] * rows
               cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
               #text = "Label: {}".format(classId)
               #cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	#	0.7, (0, 0, 255), 2)
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

