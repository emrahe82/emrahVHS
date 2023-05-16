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


def prepare_for_inference(img,target_size,segmentation_model):
  segmentation_results=segmentation_model.predict(img)
  a=segmentOutImageMinimalBorder(img,segmentation_results[0])
  a=convert_to_square_image(a)
  a=cv2.resize(a, target_size, interpolation=cv2.INTER_LINEAR)
  return a
