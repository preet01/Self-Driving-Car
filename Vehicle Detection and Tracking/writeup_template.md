
**Vehicle Detection Project**

The goals / steps of this project are the following:

* I was provided with a vehicle vs non-vehicle datasets, hence I choose to use all the images for prediction. Here we used two techniques to extract out the information out of the image. First we used HOG or Histogram of Gradient and second we used Spatial Bining of Color. We combined both techniques to get the best results. The histogram of gradient function comes in built with sklearn where you can chose among various setting like normalization over a cell, how much pixels to choose and how much bins of histogram do we reqire. 

* I used SVM or Support Vector machines to process these combined feature and created labels using np.ones for the correspoding values. I also normalized the features using Standard scalar from sklearn. Using this, I got around 98% accuracy.

* Sliding Search's main algorithm is selecting two x1,y1 and x2,y2 values which can represent a rectangle. This rectangle is feed into the SVM for prediction. If the result is True or 1, we append the boxes/rectangle. There can be window overlap as objects tends to be not regularly spaced in real world. 

* Sliding Search does result in many ovelapping windows which are removed if they represent the same object with the help of heatmap, where heatmap tends to represent the overall pixel value. If it's less than the threshold we delete the image as a false positive otherwise we use `scipy.ndimage.measurements.label()` which helps us to choose the box values. 


