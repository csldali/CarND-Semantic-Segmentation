# A project about Semantic segmentation

[image1]: img/NB_uu_000001.png "Case A without brightness"
[image2]: img/YB_uu_000001.png "Case A with brightness"
[image3]: img/NB_um_000067.png "Case B without brightness"
[image4]: img/YB_um_000067.png "Case B with brighness"

Semantic segmentation enhances the performance of self-driving cars by narrowing down the area through which they can drive by. Fully convolutional networks (FCN) are the models that have the best perfomance to do this kind of task, and the difference among other classfication networks is that the output is an image.

In this project, the VGG-16 model is modified so that the fully connected layers are replaced by transposed convolutional layers.

## Code explanation
This is a perfect example of *transfer learning* in which we use a pretrained model to perform another task, and during its training phase the weights of the most external layers will change slightly. Basically several functions have to be defined:
 * `layers` creates the missing layers to transform the VGG model to a FCN.
 * `optimize` where the model is set up. The loss function, the learning rate and so on.
 * `train_nn` trains the model.
 
The parameters are self-explanatory. `keep_prob` and `learning_rate` have been set to 0.5 and 0.0005 respectively. The number of epochs has not been changed.

Due to hardware capacity the current model has been run in an EC2 instance. Concretely running with the [deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GF11NF). For that reason it has been possible to feed 64 images in every batch.

## Segmentation tests

Three cases have been implemented, and in two of them **data augmentation has been applied**. The way data augmentation works is quite simple, and it is only necessary to modify `gen_batch_function` in `helper.py`:

```python
def gen_batch_function(data_folder, image_shape, brightness = False):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :param brightness: If True the batch is composed with half of the images with random brightness
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        if brightness:
            # The batch size is shrink by half if brightness
            batch_size = batch_size // 2
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

                if brightness:
                    # TODO: Change image colorspace to HSV
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    # TODO: Generate random brightness modifier
                    br = np.random.random_integers(low=-20, high=20)
                    # TODO: Modify the brightness of the image
                    hsv_image[:,:,2] = cv2.add(hsv_image[:,:,2], int(br))

                    images.append(hsv_image)
                    gt_images.append(gt_image)
                    
            yield np.array(images), np.array(gt_images)
    return get_batches_fn
```
This function implements random brightness to an image if `brightness = True`. If that flag is active, instead of passing the batch with a number of images set in the variable `batch_size` only half of them are passed as raw as they are loaded from the data folder. **The remaining half are the raw images with random brightness**.

## Results

The performance of the model is evaluated for two images with different light conditions:

**Update [2018-03-30]**: To improve the classification of pixels several parameters values have been changed:

* Batch size = 4
* Number of epochs = 30
* Learning rate = 0.00001

| Model parameters      	|     Segmentation	        					| 
|:---------------------:|:---------------------------------------------:| 
| brightness FALSE		| ![alt text][image1]			| 
| brightness TRUE		| ![alt text][image2]			| 


| Model parameters      	|     Segmentation	        					| 
|:---------------------:|:---------------------------------------------:| 
| brightness FALSE		| ![alt text][image3]			| 
| brightness TRUE	| ![alt text][image4]			| 


Overall, the data augmentation technique improves the true positive rate (TPR) of road pixels. It is clear that in poor light conditions the model is having some issues to segment the road. In these cases it might be good to try another data augmentation technique so that the images with nice light conditions become more similar to the images dark road segments.

The images in the `runs` folder correspond to the model with brightness TRUE & batch size 4.

**WARNING:** Using the AMI from Amazon requires to install the following libraries (that are currently not included) to import OpenCV
* libstdc++.so.6
* libgomp.so.1
