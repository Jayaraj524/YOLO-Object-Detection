# Real-Time Object Detection using YOLOv8 and YOLOv11

This project uses YOLOv8 or YOLOv11 to detect missing products such as Coca-Cola or Fanta bottles from supermarket shelves by analyzing YouTube videos. The model is trained using datasets from Roboflow and executed with Google Colab.

## Project Overview

- Object detection using YOLOv8 or YOLOv11
- Trained with custom datasets from Roboflow
- PyTorch model used for inference
- Detects objects in YouTube videos based on the trained dataset

## Steps to Train the Model

1. Go to universe.roboflow.com and search for the dataset you want to use
2. After selecting the dataset, choose the Ultralytics export option
3. Click on the Open in Colab button to open a training notebook
4. In Google Colab:
   - Run the first cell as it is
   - Paste the copied training code into the second cell
   - Run the second cell and wait for training to complete (usually 100 epochs)
5. Once training is done, go to the Deploy section in the Ultralytics Hub
6. Download the PyTorch (.pt) model file from there

## How to Use This Project

1. Clone or download this repository
2. Place the downloaded .pt model file in the project directory
3. Open the Python script and update:
   - The path to the .pt model file
   - The YouTube video URL you want to analyze
4. Run the script
5. The model will detect objects in the video based on your trained dataset
