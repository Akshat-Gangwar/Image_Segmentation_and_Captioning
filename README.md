# Image_Segmentation_and_Captioning
  This is a Deep learning project I made during an internship.
  It takes input images and provides captions for it describing what is happening in the image and it segments the subjects in the image
  
  Guide to Use:
  1) Run Image_Caption_train.ipynb on kaggle
  2) Save the output model in a folder named "saved_models" as "image_captioning_coco_weights" & output 'vocab_coco.file' in a folder named "saved_vocabulary"
  3) Run image-segmentation.ipynb on kaggle
  4) Save this model in saved_models folder as "segmentation_model"
  5) Run app.py using streamlit


*I have also made a version of this which uses pretrained RESNET50 for image segmentation.
 It also marks objects in a box and gives indivisual masks for each object.
Guide to Use: 
1) Run Image_Caption_train.ipynb on kaggle
2) Save the output model in a folder named "saved_models" as "image_captioning_coco_weights" & output 'vocab_coco.file' in a folder named "saved_vocabulary"
3) Run Pretrain_app.py using streamlit




https://github.com/user-attachments/assets/20e409f0-ae28-4511-9da4-bb694d3c2c5a

