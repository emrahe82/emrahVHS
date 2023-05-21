from keras.applications import Xception
from keras import layers as lyr
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2
import requests
import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image as PPPImage
from io import BytesIO

BASE_LINK_FOR_VHSCOLLECT_IMAGES = "https://vhscollector.com/sites/default/files/vhsimages/"
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
  segmentation_results=segmentation_model.predict(img)
  a=segmentOutImageMinimalBorder(img,segmentation_results[0]) #crop out the segmented image
  if(a is None):
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
