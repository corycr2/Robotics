import cv2
import numpy as np
import os

# Load all images from the folder
images = []
folder = "/Users/corylewis/Desktop/Robotics/first_200_right"
for filename in sorted(os.listdir(folder)):
    print(filename)
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        images.append(img)


# Define the SIFT detector
sift = cv2.SIFT_create()

# For each image, detect the keypoints and compute their descriptors
keypoints_list = []
descriptors_list = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('Cory_Lewis_Homework1_video.avi', fourcc, 20.0, (img.shape[1], img.shape[0]))

# For each image, draw lines between the matched keypoints and write the frame to the video
last_descriptors = None
last_keypoints = None
bf = cv2.BFMatcher()
for i, (img, keypoints, descriptors) in enumerate(zip(images, keypoints_list, descriptors_list)):
    #Draw keypoints in the frame in blue using the default flag which is just a circle
    img_keypoints = cv2.drawKeypoints(img, keypoints, 0, (255,0,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    #This just adds some text to the frames stating the frame number and my name
    cv2.putText(img_keypoints, "Cory Lewis Frame {}".format(i), (900,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0), 2)
    if last_descriptors is not None:
        matches = bf.knnMatch(descriptors, last_descriptors, k=2)

        #create a list for good matches
        good_matches = []

        #For best match and second best match test if their distance is less than .3 of the origianl frame
        #This is to limit matches that do not correlate 
        for best, second_best in matches:
            if best.distance < .3 * second_best.distance:

                #add the best match to the list good_matches
                good_matches.append(best)

         #for best match draw a red line from the x and y coordinates in one image to the new x and y coordinate in the next image       
        for best in good_matches:
            ximg1, yimg1 = keypoints[best.queryIdx].pt
            ximg2, yimg2 = last_keypoints[best.trainIdx].pt
            img_keypoints = cv2.line(img_keypoints, (int(ximg1), int(yimg1)), (int(ximg2), int(yimg2)), (0,0,255), 2)

    #update the descriptors and keypoints to the previous descriptors and keypoints        
    last_descriptors = descriptors
    last_keypoints = keypoints

    #write image with the drawn line to our video as a frame
    video.write(img_keypoints)

#close the video and windows
video.release()
cv2.destroyAllWindows()
