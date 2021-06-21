# Handwritten-digits-recognition-by-SVM-classifier
**The objective of this project is to use the Support Vector Machine (SVM)
classifier to train a simple system for the recognition of handwritten digits
(0, 1, …, 9).**

This project uses the MNIST database of handwritten digits for testing and
training the system which is attached here as "mnist_handwritten.csv." It
contains 60k images that are sorted as one vector, each with its labels.

To begin with, we select a subset of the database consisting of the first 1000
images (image indexes from 0-1000) of handwritten digits for training the system
and use another 100 images for testing the system (image indexes from
2000-2100). Then we train the SVM classifier with the training set and classify
the handwritten digits of the testing images using the trained SVM model.
Finally, we can compute the accuracy.

First, I read the MNIST dataset file and put it in a Pandas DataFrame. I
converted the dataset to a CSV file to speed up this process and then put the
CSV file to a DataFrame. Then, I separated the train set and test set by
considering the image label column and image data columns. Finally, using
function svm_classifing, I generated the predicted labels for the test set data
and calculated the confusion matrix and the accuracy by functions
confusion_matrix and classification_report separately. As you can see in the
figure below, the accuracy is 85%.

**![](media/4ffb76e9cf93b6673fc4aec7b5c6cd4a.png)**

We repeat the experiment for training the SVM classifier with a large database.
The first 10000 images for training (image indexes from 0-10000) and test with
100 images (images indexes from 20000-20100).

As you can see in the figure below, we can achieve a more accurate model by
increasing the size of the training set. Having 10000 images in the training
set, the accuracy is 90%.

**![](media/774cef958349a826c1bb41ab1b3a17bb.png)**

By repeating the experiment for training the SVM classifier with the same set of
training images (image indexes from 0-10000) and test with another 1000 images
(image indexes from 20000-21000), considering a decent size for the test set, we
can find our model's maximum accuracy. Here, increasing the test set's size does
not significantly affect calculating the accuracy because it is the utmost
accuracy.

**![](media/6a6f8094d729a91f1d17d7fbc9068eea.png)**

We depicted the confusion matrices in the figures above.

To visually compare the classifier's accuracy, six images and their
corresponding predicted numbers are randomly selected. As the figure below
shows, all the predictions are correct.

| ![](media/c1412d3ed8e9849ce0868a8aefcb27e0.jpg)                | ![](media/e6d8fce761070d5200fd294781dc59d3.jpg) |
|----------------------------------------------------------------|-------------------------------------------------|
| ![](media/221c928c1ff2f294dd8b09690858ed85.jpg)                | ![](media/bfd35da10062bbe58a0f89f5c86ab38c.jpg) |
| ![](media/b133ad66bcddf8e0f6647f35881faf5e.jpg)                | ![](media/b9c638906586368c82eb036319025be3.jpg) |
| Predicted number and the actual number for the original images |                                                 |

We can now repeat the experiment for training the SVM classifier with different
kernel functions (e. g. rbf, polynomial, etc.).

Applying different kernels, we can find that the Gaussian kernel (rbf) has the
best accuracy (95%). One reason is that in the MNIST dataset, the non-black
pixels have a Gaussian distribution.

**![](media/be2734d9a619db57ba2dc5f8a5c11658.png)**

**![](media/1c4fb536c7bf520bfec81ec1ac630c47.png)**

**![](media/5d2a9adc96817c0f8dcfd6600656dce6.png)**

Then we create binary images of the handwritten digits by simple thresholding
and repeat the experiment above.

Since converting gray images to binary adds some noise to the image or
eliminates some parts of the image, it cannot increase the accuracy but also
decreases the accuracy in some cases. The following figures show the
classifier's evaluation of binary images with a threshold of {200, 150, 100, 50,
1}. The best results are for threshold 200.

**![](media/2f5c5ed67b2c9e3e28f52fe255dfd840.png)**

| ![](media/c53520f2e5dd460faa502fab5cbbe1b3.jpg)               | ![](media/c5392b87489609374d8e2340c5e25c28.jpg) |
|---------------------------------------------------------------|-------------------------------------------------|
| ![](media/8d85761fc5919bad4c903f138331b720.jpg)               | ![](media/20f7191de5e7db96d5327d419c55a16c.jpg) |
| ![](media/7cb71848fab7d31db02679fa3e29b01e.jpg)               | ![](media/15c1662f7a82f3bbc8c8493c3ffdb5bf.jpg) |
| The predicted number and the actual number when threshold=200 |                                                 |

**![](media/644fa516d10abfd4ed80026b39dca61a.png)**

| ![](media/e566efa7ffde29831891dc158ce26f66.jpg)               | ![](media/86affa4616797a7d559c7bdb1817f1b1.jpg) |
|---------------------------------------------------------------|-------------------------------------------------|
| ![](media/d03755ba29cee64c0da160e96d865532.jpg)               | ![](media/368cf14e45b2ef45955c8f9436b40d87.jpg) |
| ![](media/d134592c7d0e8747f5421b4853a5412f.jpg)               | ![](media/9ec279cbb61b8fd221521ea3f5d8a90c.jpg) |
| The predicted number and the actual number when threshold=150 |                                                 |

**![](media/bd19ae14319b8dd31508a2d120ff6174.png)**

| ![](media/d7751fe91f5c029cbb257147d000d341.jpg)               | ![](media/5a4faea4298fe6d2267051cf5c639862.jpg) |
|---------------------------------------------------------------|-------------------------------------------------|
| ![](media/7606a240232c983a7b59aaa58b980e36.jpg)               | ![](media/12e71a23c842679d08919a4a1727544b.jpg) |
| ![](media/8c5fbe136fa23b5fdfacaac0b4235ef1.jpg)               | ![](media/027b5af9179e7e254db98b7917e4e449.jpg) |
| The predicted number and the actual number when threshold=100 |                                                 |

**![](media/8a2bdadc4218db30d9c6b43179a8b74a.png)**

| ![](media/1f3f388c9d27fc8d78170603f75ead3e.jpg)              | ![](media/4a260eb88de0eab675d4005d8c086f78.jpg) |
|--------------------------------------------------------------|-------------------------------------------------|
| ![](media/e6c5a2a43d0fb9954fecce6533a830f0.jpg)              | ![](media/c144f78a62dd7d534186b3ce0acc1bee.jpg) |
| ![](media/029594da57f3878effc8754382ffae05.jpg)              | ![](media/59b4de759c8c8977abab64e48708830a.jpg) |
| The predicted number and the actual number when threshold=50 |                                                 |

**![](media/d56a14e0d8e8e23e6a28b308d608027d.png)**

| ![](media/48dd2f88f0b5bc0dfa80ecfd1b220e7a.jpg)             | ![](media/b44caa180f523aa0cea268fa7a820620.jpg) |
|-------------------------------------------------------------|-------------------------------------------------|
| ![](media/6636ca6d235dcd9cd659b130222db6c1.jpg)             | ![](media/6d513ad6f7e9e60e8cbc7f7c2194c5b0.jpg) |
| ![](media/0b362f635c6f2f4bf8aa76c23cc6f718.jpg)             | ![](media/217083bea9851d316ca58a450081adf5.jpg) |
| The predicted number and the actual number when threshold=1 |                                                 |

The table below is a comparison table for the accuracy of all parts above.

| Dataset size                     | Method | Accuracy |
|----------------------------------|--------|----------|
| Training set 1000 Test set 100   | Linear | 0.85     |
| Training set 10000 Test set 100  | Linear | 0.90     |
| Training set 10000 Test set 1000 | Linear | 0.91     |

*Comparing the effect of training set's size on the accuracy*

| Dataset size                     | Method  | Accuracy |
|----------------------------------|---------|----------|
| Training set 10000 Test set 1000 | Linear  | 0.91     |
| Training set 10000 Test set 1000 | Poly    | 0.79     |
| Training set 10000 Test set 1000 | rbf     | 0.95     |
| Training set 10000 Test set 1000 | sigmoid | 0.81     |

*Comparing the effect of the kernel on the accuracy*

| Dataset size                                                             | Method | Image type                                      | Accuracy |
|--------------------------------------------------------------------------|--------|-------------------------------------------------|----------|
| Training set 10000 Test set 1000                                         | Linear | Original                                        | 0.91     |
| Training set 10000 Test set 1000                                         | Linear | Binary – Thr 200                                | 0.87     |
| Training set 10000 Test set 1000                                         | Linear | Binary – Thr 150                                | 0.88     |
| Training set 10000 Test set 1000                                         | Linear | Binary – Thr 100                                | 0.89     |
| Training set 10000 Test set 1000                                         | Linear | Binary – Thr 50                                 | 0.88     |
| Training set 10000 Test set 1000                                         | Linear | Binary – Thr 1                                  | 0.90     |
| *Comparing the effect of binary threshold on the accuracy*               |        |                                                 |          |
| ![](media/b9c638906586368c82eb036319025be3.jpg)                          |        | ![](media/15c1662f7a82f3bbc8c8493c3ffdb5bf.jpg) |          |
| Original                                                                 |        | Threshold=200                                   |          |
| ![](media/9ec279cbb61b8fd221521ea3f5d8a90c.jpg)                          |        | ![](media/027b5af9179e7e254db98b7917e4e449.jpg) |          |
| Threshold=150                                                            |        | Threshold=100                                   |          |
| ![](media/59b4de759c8c8977abab64e48708830a.jpg)                          |        | ![](media/217083bea9851d316ca58a450081adf5.jpg) |          |
| Threshold=50                                                             |        | Threshold=1                                     |          |
| Comparing the effect of the binary images on the accuracy of the trainer |        |                                                 |          |

**References:**

<https://towardsdatascience.com/support-vector-machine-mnist-digit-classification-with-python-including-my-hand-written-digits-83d6eca7004a>

<https://pjreddie.com/projects/mnist-in-csv/>

<https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/>

<https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8>

