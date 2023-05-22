from keras.applications import Xception
from keras import layers as lyr
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2
import requests
import os
import io
from io import BytesIO
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image as PPPImage
from io import BytesIO
import ultralytics
from ultralytics import YOLO 
import pandas as pd
from google.colab.patches import cv2_imshow
import urllib.request
import openpyxl #need for reading xlsm
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment
import imagehash
from typing import Tuple
import pickle
import sys
import copyreg

BASE_LINK_FOR_VHSCOLLECT_IMAGES = "https://vhscollector.com/sites/default/files/vhsimages/"
BACKUP_IMAGE_URL_BASE_PATH = "/content/drive/MyDrive/fiverr/rhyanschwartz/VHScollector images download/"
WORK_WITH_URL = False 


#Reads OCR from the given image path
def grabOCR(IMAGE_PATH):
    # Check if the image file exists
    if not os.path.isfile(IMAGE_PATH):
        raise ValueError(f"The file '{IMAGE_PATH}' does not exist.")

    # Load the image file
    with io.open(IMAGE_PATH, 'rb') as image_file:
        content = image_file.read()

    # Create a Vision API image object
    image = types.Image(content=content)

    # Detect text in the image
    try:
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
    except Exception as e:
        raise ValueError(f"Error detecting text in image - Vision AI: {e}")
        return "",""

    # Print the detected text
    output = ""
    for text in texts:
        output += text.description + " "
    return output.strip(),response
    
#Splits the OCR according to given x coordinates
#if a combined image was fed to the OCR (to save tokens)
def splitOCRAccordingtoCoordinates(response,x_coords):
  n=len(x_coords)-1
  OCRTextPartialLists=[[] for _ in range(n)]
  #print(OCRTextPartialLists)

  for i,annotation in enumerate(response.text_annotations):
    if(i>0):
      #print(annotation.description)
      vertices = np.array([(v.x, v.y) for v in annotation.bounding_poly.vertices], dtype=np.int32)
      center = tuple(vertices.mean(axis=0).astype(np.int32))
      #print(center)
      index = next((i for i, x in enumerate(x_coords) if x > center[0]), len(x_coords))-1
      #print(index)
      if(index>(n-1)):
        index=n-1
      elif(index<0):
        index=0
      (OCRTextPartialLists[index]).append(annotation.description)
  return OCRTextPartialLists



#this function reads from google drive backup images
#instead of vhscollector.com . if fails, tries from the url anyway
def readImagesFrom_googledrive_backup_images(imageLinks):
  images = []
  heights = []
  widths = []
  for i,link in enumerate(imageLinks):
    vhs_id = (link.replace(BASE_LINK_FOR_VHSCOLLECT_IMAGES,"")).split("_")[0]
    print("vhs_id: " +str(vhs_id))
    if(len(vhs_id)>8):
      print("different style of url! this probably not stored in drive. using the url")
      try:
        imgy = read_image_from_url(link)
        images.append(imgy)
        heights.append(imgy.shape[0])
        widths.append(imgy.shape[1])
      except:
        print("couldn't read from url too!")
    else:
      new_image_path_from_backup = BACKUP_IMAGE_URL_BASE_PATH+str(vhs_id)+"_"+str(i)+".jpg"
      print(new_image_path_from_backup)
      try:
        imgy = cv2.imread(new_image_path_from_backup)
        images.append(imgy)
        heights.append(imgy.shape[0])
        widths.append(imgy.shape[1])
      except:
        print("couldn't read: "+str(new_image_path_from_backup))
  return images, heights, widths



#this function reads image from the url
def read_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for 4xx and 5xx status codes
        img = PPPImage.open(BytesIO(response.content))
        img.load() # Load the image data into memory
        if(img):
          img_array = np.asarray(img).clip(0, 255).astype(np.uint8)
          img_array = np.flip(img_array, axis=-1)
          #cv2_imshow(img_array)
        return img_array

    except (requests.exceptions.RequestException, OSError) as e:
        print(f"Failed to read image from URL: {e}")
        return None
    

                    
#add specific sized black border around input iamge
def add_black_border(image, border_size=20):
    if image is None:
        return None

    # Add a black border of specified size
    border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=0)
    return border_image


#The function hConcatImageArray(imageLinks) is designed to take in a list of image links (either URLs or file paths to images located in Google Drive, 
#depending on the setting of WORK_WITH_URL), and horizontally concatenate (i.e., stitch together side by side) all these images into one single image.
#Also returns the x coordinates, where they are stiched
def hConcatImageArray(imageLinks):
    # Read images from URLs and get their widths and heights
    if(WORK_WITH_URL):
      images, heights, widths = readImagesFromURLs(imageLinks)
    else:
      images, heights, widths = readImagesFrom_googledrive_backup_images(imageLinks)

    # Compute the x coordinates of the concatenations
    x_coords = np.cumsum([0] + widths[:-1])
    
    # Create a black canvas with the total height and width
    try:    
      total_height = max(heights)
      total_width = sum(widths)
      canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    except:
      return None,None  

    # Paste the images onto the canvas at the appropriate x coordinates
    for i, img in enumerate(images):
      if(img is not None):
        x, y = x_coords[i], 0
        if(img.shape[2]==4):
          canvas[y:y+img.shape[0], x:x+img.shape[1], :] = img[:,:,0:-1] #incase of png
        else:
          canvas[y:y+img.shape[0], x:x+img.shape[1], :] = img #incase of non png
    try:
      x_coords = np.concatenate((x_coords, [total_width-1]))
      return canvas, x_coords
    except:
      print("couldnt read the image Links!")
      for i in range(imageLinks):
        print(i)
      return None,None
    

#This function takes in YOLO flat tape segmentation results and a maximum x value as input. 
#It extracts the x-coordinates of the bounding boxes from the segmentation results, 
#sorts them in ascending order, and stores them in an array. 
#The function then checks for consecutive x-values that have a difference of less than 5% and computes their average. 
#If the average value falls outside the range of 0 to max_x, it is clipped to that range. 
#The final averaged x-coordinates are stored in an array and returned by the function.
def get_x_coords_from_yolo_segmentation_results(results, max_x):
  # Get the tensor of bounding box coordinates
  boxes = results[0].boxes.data

  # Initialize a list to store x-coordinates
  x_coords = []

  # Iterate over the rows of the tensor and extract x-coordinates
  for i in range(boxes.shape[0]):
      x1 = int(boxes[i][0].item())
      x2 = int(boxes[i][2].item())
      x_coords.extend([x1, x2])

  # Sort the array in ascending order
  x_coords_sorted = np.sort(x_coords)

  # Initialize a list to store the final x-coordinates
  final_x_coords = []

  # Initialize variables to store the start and end index of the current group of values
  start_idx = 0
  end_idx = 0

  # Loop through the sorted array
  while end_idx < len(x_coords_sorted):
      # Check if the difference between the current and next value is less than 5%
      if end_idx+1 < len(x_coords_sorted) and (x_coords_sorted[end_idx+1] - x_coords_sorted[end_idx])/x_coords_sorted[end_idx] < 0.05:
          # If so, increment the end index and continue
          end_idx += 1
      else:
          # If not, compute the average of the values in the current group and append to the final list
          avg_val = int(np.mean(x_coords_sorted[start_idx:end_idx+1]))
          avg_val = max(min(avg_val,max_x),0)
          final_x_coords.append(avg_val)
          # Reset the start and end index for the next group
          start_idx = end_idx + 1
          end_idx = start_idx

  # Print the resulting array
  return final_x_coords



#input is a random size image, and makes it square by adding 
#black contours around the image
def convert_to_square_image(img):
    height, width = img.shape[:2]

    # Calculate the size of the square image
    size = max(width, height)

    # Create a new image with a white background
    square_img = np.zeros((size, size, 3), dtype=np.uint8)
    square_img.fill(0)
    
    # Calculate the position to paste the input image in the center of the square image
    x = (size - width) // 2
    y = (size - height) // 2
    
    # Paste the input image into the center of the square image
    square_img[y:y+height, x:x+width] = img
    return square_img

  
  
#crops out the image  
#works only if image is frontvhs, else return none!
def segmentOutImageMinimalBorder(img,yolo_results):
  img2 = np.ascontiguousarray(img, dtype=np.uint8)
  mask=img2[:,:,0]*0

  cover_types=yolo_results.names
  for i,boxy in enumerate(yolo_results.boxes.data):
    detected_type = cover_types[int(boxy[-1].item())]
    if(detected_type=="frontvhs"):
      segs=(yolo_results.masks[i].xy)[0]  
      segsNew = segs.reshape((-1, 1, 2))
      segsNew = segsNew.astype(np.int32)
      mask = cv2.fillPoly(mask,[segsNew],(255))

      mb = cv2.bitwise_and(img2[:,:,0], mask)
      mg = cv2.bitwise_and(img2[:,:,1], mask)
      mr = cv2.bitwise_and(img2[:,:,2], mask)
      img3 = cv2.merge((mb, mg, mr))

      x1,y1,x2,y2,p,t=(yolo_results.boxes.data.cpu())[0].numpy()
      cropy=img3[int(y1):int(y2),int(x1):int(x2),:]
      return cropy
  return None


#prepares the image for the siamese
def prepare_for_inference(img,target_size,border_size,segmentation_model):
  img=add_black_border(img, border_size) #first add a black border around, maybe needed for vhscollector images. for increasing segmentation accuracy
  img=convert_to_square_image(img)
  segmentation_results=segmentation_model.predict(img)
  a=segmentOutImageMinimalBorder(img,segmentation_results[0]) #crop out the segmented image
  if(a is None):
      print("nothing cropped out from segmentation!!")
      return None
  else:
      a=convert_to_square_image(a)
      a=cv2.resize(a, target_size, interpolation=cv2.INTER_LINEAR)
      return a


    
#siamese model with trained weights, and extractor
def create_and_load_siamese_model_with_extractor(input_shape, model_weights_path=None):
    
    # Load the pretrained model
    # we will later replace imagenet
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    #Some unfrozen weights on the Xception model
    trainable = False
    for layer in base_model.layers:
        if layer.name == "block14_sepconv1":
            trainable = True
        layer.trainable = trainable
    
    # Add a GlobalAveragePooling2D layer after the base model
    gap = lyr.GlobalAveragePooling2D()

    # Define the Siamese network
    input_left = lyr.Input(input_shape)
    input_right = lyr.Input(input_shape)

    # Pass the inputs through the base model and GlobalAveragePooling2D layer
    encoded_left = gap(base_model(input_left))
    encoded_right = gap(base_model(input_right))

    # Compute the L1 distance between the encoded outputs
    l1_distance = lyr.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_left, encoded_right])

    # Add some dense layers for classification
    x = lyr.Dense(256, activation='relu')(l1_distance)
    x = lyr.Dropout(0.5)(x)
    x = lyr.BatchNormalization()(x)

    x = lyr.Dense(128, activation='relu')(x)
    x = lyr.Dropout(0.5)(x)
    x = lyr.BatchNormalization()(x)

    x = lyr.Dense(128, activation='relu')(x)
    x = lyr.Dropout(0.5)(x)
    x = lyr.BatchNormalization()(x)

    predictions = lyr.Dense(1, activation='sigmoid')(x)

    # Define the Siamese network
    siamese_model = Model(inputs=[input_left, input_right], outputs=predictions)

    # Loading the trained weights
    if model_weights_path:
        siamese_model.load_weights(model_weights_path)
    
    #feature extractor
    # Define a new input layer for a single image
    input_single = lyr.Input(input_shape)

    # Pass the input through the base model and GlobalAveragePooling2D layer
    encoded_single = gap(base_model(input_single))

    # Define the feature extraction model for a single image
    feature_extractor_single = Model(inputs=input_single, outputs=encoded_single)

    return siamese_model, feature_extractor_single



#hash checker for a "single" image
#this will be used for speed ups and comnbined model
#before running it always use correct_perspective_from_contour()
def hashFinder(arr0_uint8):
    # Create PIL image
    pil_image_0 = PPPImage.fromarray(arr0_uint8)

    # Calculate phash
    phash_0 = imagehash.phash(pil_image_0)

    # Calculate dhash
    dhash_0 = imagehash.dhash(pil_image_0)

    return phash_0, dhash_0


def siftFinder(arr0_uint8):
    #grayscale the image and find sift
    img0_gray = cv2.cvtColor(arr0_uint8, cv2.COLOR_RGB2GRAY)
    sift_0 = cv2.SIFT_create()
    # Detect keypoints and compute descriptors for both images
    sift_keypoints_0, sift_descriptors_0 = sift_0.detectAndCompute(img0_gray, None)
    return sift_keypoints_0,  sift_descriptors_0



#Detect the rectangle in the image for perspective correction.
#:param img: Input image
#:return: Tuple of list of corner points if a rectangle is detected, the image with the drawn initial contour and the image with the drawn approximated contour, else None
#Feed output to the correct_perspective_from_contour()
def detect_rectangle_via_contour(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get non-zero pixels
    _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    if not contours:
        print("No contour detected, cannot correct perspective.")
        return None, img, img

    # Copy image to draw the initial contour
    img_initial_contour = img.copy()

    # Draw the initial contour on the image
    cv2.drawContours(img_initial_contour, contours, -1, (0, 255, 0), 3)

    # Approximate contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    while len(approx) > 4:
        epsilon += 0.01
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
    while len(approx) < 4:
        epsilon -= 0.01
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Draw the approximated contour on the image
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

    return approx.reshape(-1, 2)  # reshape for the correct_perspective function


#Compute perspective transformation and apply it to the image.
#:param img: Input image
#:param corners: List of corner points for perspective correction
#:return: Perspective corrected image
#This function is required for better phash and dhash results
def correct_perspective_from_contour(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if corners is None:
        print("Not enough corners detected, skipping perspective correction.")
        return img

    # Order corners (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype = "float32")
    s = corners.sum(axis = 1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis = 1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    # Compute the new image dimensions
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Compute the perspective transformation matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))





#This function is used to extract features from a given row of a database, and add them to a dictionary for future reference.
#Parameters:
#current_input_row (int): The row index in the combined_database dataframe that we want to process.
#movieDictionary (dict): A dictionary where we will store the features for each movie.
#combined_database (DataFrame): A Pandas dataframe that contains data about movies, including links to images.
#flat_image_split_model (model): The model used to split flat images.
#segmentation_model (model): The model used for image segmentation.
#target_size (tuple): The target size for the images to be processed.
#input_shape (tuple): The input shape that the siamese network expects.
#siamese_extractor (model): The siamese model used to extract features from the images.
#Returns:
#movieDictionary (dict): The updated dictionary with the features of the current_input_row added.
def extractFeaturesfromDataBase(current_input_row, movieDictionary, combined_database, flat_image_split_model, segmentation_model, target_size, input_shape, siamese_extractor):
    imgArray=[]
    if (current_input_row in movieDictionary):
      print("already have the item!")
    else:
      try:
        imageLinks=(combined_database.iloc[current_input_row])["Images"].split(",")
        print(imageLinks)
      except:
        print("failed obtaining image links!, probably empty!")

      #EXTRACTING IMAGES from the LINKS
      numImageLinks=len(imageLinks)
      if(numImageLinks==0):
        print("NO LINKS!!")
      if(numImageLinks==1):
        print("there is a combined image on the database, splitting with yolo segmentation!")
        print(imageLinks)
        combined = read_image_from_url(imageLinks[0])

        if(combined is None):
          print("problem with the images! : " + str(imageLinks))
          print("bypassing current row!!")
          raise SystemExit
        if(combined.shape[2]==4):
          combined=combined[:,:,0:3]

        height=combined.shape[0]
        width=combined.shape[1]
        results=flat_image_split_model.predict(combined, save=True, boxes=True, show=False, line_thickness=4)
        x_coords = get_x_coords_from_yolo_segmentation_results(results, (width-1))

        for j,x in enumerate(x_coords):
          if(j<(len(x_coords)-1)):
            img=combined[:,x_coords[j]:x_coords[j+1],0:3]
            imgArray.append(img)   

      else:
        for j,link in enumerate(imageLinks):
          img = read_image_from_url(link)
          imgArray.append(img)


      for i,img in enumerate(imgArray):
        imgReadForInference=prepare_for_inference(img,target_size,20,segmentation_model)

        if(imgReadForInference is None):
          phash_, dhash_, sift_keypoints_,  sift_descriptors_  = None, None, None, None

        else:
          cv2_imshow(imgReadForInference)

          corners_ = detect_rectangle_via_contour(imgReadForInference)
          corr_arr_ = correct_perspective_from_contour(imgReadForInference, corners_)

          phash_, dhash_ = hashFinder(corr_arr_)
          sift_keypoints_,  sift_descriptors_= siftFinder(corr_arr_)

          img_array = imgReadForInference / 255.0
          img_array = np.expand_dims(img_array, axis=0)

          assert img_array.shape == (1,) + input_shape, f'Expected shape: {(1,) + input_shape}, but got: {img_array.shape}'

          siamese_features_ = siamese_extractor.predict(img_array)

        subDictionary={}
        print("adding an item to dictionary..")
        subDictionary["itemNumber"] = current_input_row
        subDictionary["movieSubImageNumber"] = i
        subDictionary["siameseFeatures"] = siamese_features_
        subDictionary["dHash"] = dhash_
        subDictionary["pHash"] = phash_
        subDictionary["siftKeypointsList"] = sift_keypoints_
        subDictionary["siftDescriptorsList"] = sift_descriptors_ 

        if current_input_row not in movieDictionary:
          movieDictionary[current_input_row] = [subDictionary]
        else:
          movieDictionary[current_input_row].append(subDictionary)
    
    return movieDictionary


#https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
#a function to make cv2.KeyPoint objects storable and 
#retrievable using Python's built-in pickle module.
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
#one time use:
#copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

