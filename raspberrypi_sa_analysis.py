import tensorflow as tf
import numpy as np
import time 
import cv2
from datetime import datetime as dt

def rectangleMask(frame, x = 360, y = 190, x2 = 874, y2 = 1000):
    height,width, c_channels = frame.shape

    mask = np.zeros((height,width,c_channels), np.uint8)

    rectangleImg = cv2.rectangle(mask, (x, y), (x2, y2),(255,255,255), thickness=-1)

    masked_data = cv2.bitwise_and(frame, rectangleImg)   
    masked_data[np.where((masked_data==[0,0,0]).all(axis=2))] = [255,255,255]

    return masked_data 

def per_image_standardization(image):
    return (image - np.mean(image)) / np.std(image)

def f_center(xmin, xmax, ymin, ymax):
    # Return center of box
    center = []
    center.append(np.average([xmin,xmax]))
    center.append(np.average([ymin,ymax]))
    return center

def imagePadder(image, dimensions):
    #Pad images to correct shape
    w, h, _ = image.shape

    n_w, n_h = dimensions

    if w > n_w or h > n_h:
        print("TOO BIG!!!!")
        return None

    n_image = cv2.copyMakeBorder(image, int((n_w - w) / 2), int((n_w - w) / 2), int((n_h - h) / 2), int((n_h - h) / 2), cv2.BORDER_CONSTANT)
    
    if n_image.shape[0] == dimensions[0] - 1:
        n_image = cv2.copyMakeBorder(n_image, 0, 1, 0, 0, cv2.BORDER_CONSTANT)

    if n_image.shape[1] == dimensions[0] - 1:
        n_image = cv2.copyMakeBorder(n_image, 0, 0, 1, 0, cv2.BORDER_CONSTANT)
        
    if n_image.shape[0] == dimensions[1] - 1:
        n_image = cv2.copyMakeBorder(n_image, 1, 0, 0, 0, cv2.BORDER_CONSTANT)
        
    if n_image.shape[1] == dimensions[1] - 1:
        n_image = cv2.copyMakeBorder(n_image, 0, 0, 0, 1, cv2.BORDER_CONSTANT)
    
    return n_image

SHOW = True

SHOW_INDICATOR = True

TIME_STAMP = 5

file_name = ".npy"

sa_interpreter = tf.lite.Interpreter(model_path="sa_model.tflite")
sa_interpreter.allocate_tensors()
_, input_height, input_width, _ = sa_interpreter.get_input_details()[0]['shape']

#Mouse Detector Initialization

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mouse_model.tflite")
interpreter.resize_tensor_input(0, [1, 640, 640, 3])
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ['mouse'] 


# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = image_path
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(input_tensor=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['num_detections']))
  scores = np.squeeze(output['detection_scores'])
  classes = np.squeeze(output['detection_classes'])
  boxes = np.squeeze(output['detection_boxes'])

  ind = np.argmax(scores)


  result = {
    'bounding_box': boxes[ind],
    'class_id': classes[ind],
    'score': scores[ind]
  }

  return result


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    obj = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    
    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return {"score":obj["score"], "bounding_box":[xmin,xmax,ymin,ymax]}

# CV2 setup
inputSource = input("Input Video?")
cap = cv2.VideoCapture(inputSource)
fps = cap.get(cv2.CAP_PROP_FPS)

def average(l):
    return sum(l)/len(l)


detectTimes = [20, 10, 5, 3, 2, 1]

numFrames = 60

count = 0

predictions = []

class SAanalysis:
    def __init__(self, diff_len, centers_len, frames_len):
        self.diffs_len = diff_len
        self.centers_len = centers_len
        self.frames_len = frames_len
        self.diffs = []
        self.centers = []
        self.frames = []

    def diffAppend(self, value):
        if len(self.diffs) < self.diffs_len:
            self.diffs.append(value)
        elif len(self.diffs) == self.diffs_len:
            self.diffs.pop(0)
            self.diffs.append(value)

    def centerAppend(self, value):
        if len(self.centers) < self.centers_len:
            self.centers.append(value)
        elif len(self.centers) == self.centers_len:
            self.centers.pop(0)
            self.centers.append(value)

    def framesAppend(self, value):
        if len(self.frames) < self.frames_len:
            self.frames.append(value)
        elif len(self.frames) == self.frames_len:
            self.frames.pop(0)
            self.frames.append(value)

    def inference(self):
        diffs_avg = np.average(self.diffs)

        n_center = []
        for i,k in zip(self.centers[0::], self.centers[1::]):
            d = np.linalg.norm(np.array(i) - np.array(k))
            n_center.append(d)
            

        centers_avg = np.average(n_center)    

        t_diffs = tf.convert_to_tensor([diffs_avg], tf.float32)
        t_centers = tf.convert_to_tensor([centers_avg], tf.float32)
        print(t_diffs)
        print(t_centers)

        t_frames = [tf.cast(tf.image.per_image_standardization(tf.image.resize(tf.convert_to_tensor(frame), (input_height, input_width))), tf.float32) for frame in self.frames]

        d = {'frames':[t_frames[0][tf.newaxis, ...],t_frames[1][tf.newaxis, ...],t_frames[2][tf.newaxis, ...],t_frames[3][tf.newaxis, ...],t_frames[4][tf.newaxis, ...],t_frames[5][tf.newaxis, ...]], 'diffs': t_diffs[tf.newaxis, ...], 'centers' :t_centers[tf.newaxis, ...]}
        return d


sa_analyser = SAanalysis(60, 30, 6)

pFrame = None

c = 1

ret = True

latest_predict = None

print("Running!")

while(ret):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", frame)

    count += 1

    
    if TIME_STAMP:
        if count % (TIME_STAMP*60*fps) == 0:
            print(f"Processing Min {count / (60*fps)}")
    if count % fps != 0:
        continue
    
    
    
    nFrame = rectangleMask(frame, x = 350, y = 110, x2 = 865,y2 = 640)
    if SHOW:
        cv2.imshow("nFrame", nFrame)

    if pFrame is not None:
        diff = cv2.absdiff(nFrame, pFrame).sum()
        sa_analyser.diffAppend(diff)
    
    pFrame = nFrame.copy()
    
    detections = run_odt_and_draw_results(nFrame, interpreter)
    
    scores = detections["score"]
    bb  = detections["bounding_box"]

    n_height, n_width, _ = nFrame.shape
    cv2.rectangle(frame, (int((bb[0])), int(bb[2])), (int(bb[1]), int(bb[3])), (0, 255, 0 ), 3) 



    print("CKPT3")
    
    center = f_center(*bb)
    sa_analyser.centerAppend(center)

    y_diff = numFrames - c

    if y_diff in detectTimes:
        nC = frame[int(bb[0]):int(bb[1]),int(bb[2]):int(bb[3])]
        nPadded = imagePadder(nC, (250, 250))
        if nPadded is None:
            c = c - 1
            continue
        nPadded = cv2.cvtColor(nPadded, cv2.COLOR_RGB2BGR)
        sa_analyser.framesAppend(nPadded)

    c += 1
    print("CKPT5")
    print(c)

    if c == numFrames:
        sa_out = sa_analyser.inference()
        frames = [0,1,2,5,6,7]
        for i,x in enumerate(frames):
            sa_interpreter.set_tensor(x, sa_out['frames'][i])
        sa_interpreter.set_tensor(3, sa_out['centers'])
        sa_interpreter.set_tensor(4, sa_out['diffs'])
        sa_interpreter.invoke()
        prediction = sa_interpreter.get_tensor(225)
        print(prediction)
        predictions.append([count, prediction])
        c = 1
        latest_predict = float(prediction)

    if SHOW:
        if SHOW_INDICATOR and latest_predict is not None:
            cv2.circle(frame, (150, 150), 20, (255 * (latest_predict - 0.6)*2, 0, 255 - (255*(latest_predict - 0.6)*2)), -1)
        cv2.imshow("Frame", frame)

    if SHOW:
        cv2.imshow("Frame", frame)
    if SHOW:
        # show one frame at a time
        key = cv2.waitKey(1)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break


print("Saving!!!")

nPredictions = np.asarray(predictions)

with open(f"predictions_{inputSource}_{file_name}",'wb') as f:
    np.save(f, nPredictions)



cap.release()
cv2.destroyAllWindows()