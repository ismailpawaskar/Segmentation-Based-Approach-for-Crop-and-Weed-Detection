# Segmentation Based Approach for Crop and Weed Detection

## Overview
This project aims to automate crop and weed detection in complex field environments using segmentation-based approaches. The primary focus is on evaluating various image segmentation models and examining the behavior of object detection algorithms. By leveraging these techniques, the goal is to develop a robust system capable of accurately identifying crops and weeds in agricultural settings, thereby assisting farmers in optimizing their crop management strategies.

## Background
The detection of crops and weeds in agricultural fields is essential for effective crop management and yield optimization. Traditional methods of manual inspection and monitoring are time-consuming and labor-intensive, making them impractical for large-scale farming operations. By employing advanced computer vision techniques, such as image segmentation and object detection, it is possible to automate this process and provide farmers with real-time insights into the health and status of their crops.

## Key Features
- **Extensive Literature Review:** Conducted an in-depth review of over 20 research papers on image segmentation models and object detection algorithms to identify state-of-the-art techniques and methodologies.
- **Model Evaluation:** Evaluated the performance of various segmentation models, including Semantic Segmentation and Instance Segmentation, in complex field environments to determine their suitability for crop and weed detection tasks.
- **YOLO Algorithm Examination:** Investigated the behavior and performance of different versions of the You Only Look Once (YOLO) algorithm for object detection in agricultural settings, considering factors such as speed, accuracy, and robustness.
- **SAM Implementation:** Implemented the Segment Anything Model (SAM) for automated crop and weed detection, leveraging its capabilities in semantic segmentation to accurately delineate objects of interest in agricultural imagery.
- **YOLOv8 Integration:** Integrated the latest version of the You Only Look Once (YOLOv8) algorithm into the detection pipeline to further improve detection accuracy and reliability, particularly in challenging scenarios with varying lighting conditions and occlusions.
- **Ground Truth Mask Generation:** Developed a methodology for automatically generating ground truth masks from JSON files containing image-specific details, streamlining the annotation process and facilitating the training of machine learning models.
- **SAM Customization:** Customized the architecture of the Segment Anything Model (SAM) using Morphological Operations such as Dilation and Erosion to enhance performance and mitigate issues such as noise and inaccurate object boundaries.

## Implementation
1. **Literature Review:** Conducted an extensive literature review of existing research papers and publications related to image segmentation and object detection in agricultural contexts to gain insights into current methodologies and best practices.
2. **Model Selection:** Identified and selected suitable segmentation models and object detection algorithms based on the findings of the literature review and their applicability to the task of crop and weed detection.
3. **Data Acquisition:** Gathered and curated a diverse dataset of agricultural imagery, including images of crops, weeds, and various environmental conditions, to train and evaluate the selected models.
4. **Model Training:** Trained the selected models using the curated dataset, fine-tuning their parameters and hyperparameters to optimize performance and achieve accurate detection results.
5. **Evaluation and Validation:** Evaluated the trained models using a separate validation dataset, assessing their performance metrics such as precision, recall, and F1-score to measure their effectiveness in crop and weed detection tasks.
6. **Integration and Deployment:** Integrated the trained models into a unified detection pipeline, incorporating preprocessing steps, post-processing techniques, and real-time inference capabilities to facilitate their deployment in agricultural field environments.
7. **Testing and Benchmarking:** Tested the deployed system in real-world scenarios, benchmarking its performance against existing methods and evaluating its reliability, accuracy, and efficiency in detecting crops and weeds under varying conditions.

# How it Works

The process flow for the segmentation-based approach for crop and weed detection involves the following steps:

1. **Image Annotation:** The process begins with the annotation of images using annotation tools such as Supervisely or Roboflow. Annotations are used to mark the regions of interest (crops and weeds) in the images, creating labeled datasets for training and evaluation.

2. **JSON File Generation:** After annotation, JSON files containing image-specific details, such as bounding box coordinates and class labels, are generated. These JSON files serve as ground truth data for training and evaluating segmentation models.

3. **SAM Process:** The annotated images are processed using the Segment Anything Model (SAM), a semantic segmentation model capable of segmenting objects in images without the need for explicit annotation. SAM analyzes the input images and generates segmented images highlighting the regions of interest (crops and weeds).

4. **YOLOv8 Detection:** The segmented images produced by SAM are then input to the You Only Look Once (YOLOv8) object detection algorithm. YOLOv8 identifies and detects the objects (crops and weeds) in the segmented images, producing bounding boxes around them.

5. **Ground Truth Image Generation:** Meanwhile, JSON files containing image-specific details are retrieved from the "JSON_Annotation_Files" folder. These JSON files are used to generate ground truth images, providing a reference for evaluating the accuracy of the segmentation and detection algorithms.

6. **Binary Mask Generation:** The segmented images produced by SAM are converted into binary masks, where pixel values represent the presence or absence of objects (crops and weeds). These binary masks are used to quantitatively assess the performance of the segmentation algorithm.

7. **Performance Metrics Calculation:** Performance metrics such as F1 Score, Intersection over Union (IoU), Dice Coefficient, and Pixel Accuracy are calculated for each segmented image and detection result. These metrics are then stored in the "Metrics Folder" for further analysis.

8. **Mean Calculation:** To compare the performance of the original, erosion, and dilation methods, the mean values of the performance metrics are calculated across all images. This provides an overall assessment of the effectiveness of each method in crop and weed detection.

9. **Repetition and Evaluation:** Steps 1 to 8 are repeated for a subset of images (10% of the dataset) selected through random sampling. This iterative process ensures robustness and reliability in evaluating the segmentation and detection algorithms.

# How it Works

This project aims to automate the process of crop and weed detection using a segmentation-based approach, leveraging the Segment Anything Model (SAM) and You Only Look Once version 8 (YOLOv8) algorithm. The workflow involves several steps, including image annotation, model processing, ground truth image generation, and performance evaluation.

## Process Flow
1. **Image Annotation:** The process begins with annotating images using tools such as Supervisely or Roboflow to label crops and weeds. This annotated data is saved in JSON format, providing ground truth information for training and evaluation.

2. **SAM Processing:** The annotated images are fed into the Segment Anything Model (SAM), which segments the image into regions corresponding to crops and weeds. SAM operates on a single image at a time, processing each image independently.

3. **Prompt and YOLOv8 Detection:** The segmented images are then processed using the YOLOv8 algorithm for object detection. YOLOv8 identifies and detects crops and weeds within the segmented regions, providing bounding box coordinates for each detected object.

4. **Generation of Ground Truth Image:** The ground truth image is generated from the JSON files containing annotation details. Each JSON file corresponds to a single annotated image, and the ground truth image is created by overlaying the annotated regions onto the original image.

5. **Generating Binary Mask:** The segmented image is converted into a binary mask, where pixels corresponding to crops and weeds are assigned a value of 1, while background pixels are assigned a value of 0.

6. **Performance Metrics Calculation:** Performance metrics such as F1 Score, Intersection over Union (IoU), Dice Coefficient, and Pixel Accuracy are calculated by comparing the ground truth image with the binary mask. These metrics provide quantitative measures of the model's accuracy and effectiveness in detecting crops and weeds.

# Implementation

## Dataset Preparation
- Annotated images are prepared using tools like Supervisely or Roboflow, with annotations stored in JSON format.
- A subset of annotated images, representing 10% of the dataset, is randomly selected for processing due to SAM's limitation of operating on one image at a time.

## Model Processing
- Each annotated image is processed through SAM and YOLOv8 to segment and detect crops and weeds.
- Ground truth images are generated from JSON files containing annotation details, and binary masks are created from segmented images.

## Performance Evaluation
- Performance metrics including F1 Score, IoU, Dice Coefficient, and Pixel Accuracy are calculated for each processed image.
- The mean of these metrics is computed to compare the performance of the original, erosion, and dilation approaches.

## Folder Structure
- Annotated image data and JSON annotation files are stored in the "Data" directory.
- Ground truth images and binary masks are stored in the "Ground_Truth" directory.
- Performance metric results are saved in the "Metrics" directory.

