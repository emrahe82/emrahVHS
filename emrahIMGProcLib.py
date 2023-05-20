from keras.applications import Xception
from keras import layers as lyr
from keras.models import Model
from keras import backend as K
import numpy as np
import cv2
import h5py

BASE_LINK_FOR_VHSCOLLECT_IMAGES = "https://vhscollector.com/sites/default/files/vhsimages/"

#this function reads from google drive backup images
#instead of vhscollector.com
#if fails, tries url anyway
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
    
    

def check_weights_dimensions(weights_file):
    # Open the weights file.
    with h5py.File(weights_file, 'r') as f:
        # The file is organized in layers and you can navigate through it like a dictionary.
        for layer_name, layer_group in f.items():
            print("Layer:", layer_name)
            for sub_name, sub_group in layer_group.items():
                print("  - {0:<12}".format(sub_name))
                for p_name, param in sub_group.items():
                    print("    - {0:<12}: {1}".format(p_name, param.shape))


                    
#add specific sized black border around input iamge
def add_black_border(image, border_size=20):
    if image is None:
        return None

    # Add a black border of specified size
    border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=0)
    return border_image


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
