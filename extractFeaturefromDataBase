def extractFeaturefromDataBase(current_input_row, movieDictionary, combined_database, flat_image_split_model, segmentation_model, target_size, input_shape, siamese_extractor):
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
        combined = emrahIMGProcLib.read_image_from_url(imageLinks[0])

        if(combined is None):
          print("problem with the images! : " + str(imageLinks))
          print("bypassing current row!!")
          raise SystemExit
        if(combined.shape[2]==4):
          combined=combined[:,:,0:3]

        height=combined.shape[0]
        width=combined.shape[1]
        results=flat_image_split_model.predict(combined, save=True, boxes=True, show=False, line_thickness=4)
        x_coords = emrahIMGProcLib.get_x_coords_from_yolo_segmentation_results(results, (width-1))

        for j,x in enumerate(x_coords):
          if(j<(len(x_coords)-1)):
            img=combined[:,x_coords[j]:x_coords[j+1],0:3]
            imgArray.append(img)   

      else:
        for j,link in enumerate(imageLinks):
          img = emrahIMGProcLib.read_image_from_url(link)
          imgArray.append(img)


      for i,img in enumerate(imgArray):
        imgReadForInference=emrahIMGProcLib.prepare_for_inference(img,target_size,20,segmentation_model)

        if(imgReadForInference is None):
          phash_, dhash_, sift_keypoints_,  sift_descriptors_, siamese_features_ = None, None, None, None, None

        else:
          cv2_imshow(imgReadForInference)

          corners_ = emrahIMGProcLib.detect_rectangle_via_contour(imgReadForInference)
          corr_arr_ = emrahIMGProcLib.correct_perspective_from_contour(imgReadForInference, corners_)

          phash_, dhash_ = emrahIMGProcLib.hashFinder(corr_arr_)
          sift_keypoints_,  sift_descriptors_= emrahIMGProcLib.siftFinder(corr_arr_)

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
        subDictionary["siftKeypoints"] = sift_keypoints_
        subDictionary["siftDescriptors"] = sift_descriptors_

        if current_input_row not in movieDictionary:
          movieDictionary[current_input_row] = [subDictionary]
        else:
          movieDictionary[current_input_row].append(subDictionary)
    
    return movieDictionary
