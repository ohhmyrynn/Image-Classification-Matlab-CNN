## Skin Disease Classification Program Using MATLAB CNN

This program uses a convolutional neural network (CNN) to classify skin diseases from images.

This project is a classification program to detect types of skin diseases such as hives, herpes, cancer and psoriasis. 
This program was created using Matlab 2023a. 
You can use any version of Matlan to run this program.

The program documentation is still in Indonesian language format, you can use Google Translate to understand the comments in each part.

** This program is still not perfect, I have tried it by inputting 60 images per class, and 1000 epochs. 
Accuracy still shows around 70%.
Perhaps using more appropriate layers and more specific image augmentation can increase the accuracy

# Here are instructions on how you can use it:
1. Copy this repository to your local folder
2. Create a folder about the objects you want to classify in your project environment
    >> Example: Skin Diseases>>Class Subfolder
3. Enter jpg format photos in each class folder, you can adjust the number of photos in each class.
4. In the .m extension file, you can change the input code
    imds = imageDatastore('YourMainFolder',...
        'IncludeSubfolders',true,...
        'LabelSource','foldernames');
5. You can change the class name in the confusion matrix by changing classx to your object class
    classLabels = {'Class1', 'Class2', 'Class3', 'Class4'};