%% 1. Klasifikasi Penyakit Kulit menggunakan CNN
% Kelompok 3
% Alfan Umar Faruq | M0221010
% Trian Aprilianto | M0221091
clc;
clear;
close all;
rng('default'); 

%% 2. Memuat Dataset Gambar
imds = imageDatastore('PenyakitKulit',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% 3. Menampilkan Beberapa Sampel Gambar
figure
numImages = length(imds.Files);
perm = randperm(numImages,25);
for i = 1:25
    subplot(5,5,i);
    imshow(imds.Files{perm(i)});
    drawnow
end

%% 4. Augmentasi Data Gambar
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',[-180 180],...
    'RandXScale',[1 4], ...
    'RandYReflection',true, ...
    'RandYScale',[1 4]);

%% 5. Membagi Dataset menjadi Training dan Testing dengan perbandingan 6:4
[imdsTrain,imdsTest] = splitEachLabel(imds,0.6,'randomize');
%% 6. Preprocessing Gambar
%Semua gambar diubah ukurannya menjadi 64x64x3 pixel, diaugmentasi baik
%data train maupun validasi
imageSize = [64 64 3];
datastoreTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);
datastoreTest = augmentedImageDatastore(imageSize,imdsTest);
%% 7. Membangun Model Klasifikasi
layers = [ ...
    imageInputLayer(imageSize,'Name','input')
    convolution2dLayer(1,4,'Padding','same')  %adding kernel size 1x1, filter size 4
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(6,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(64)
    reluLayer    
    fullyConnectedLayer(32)
    reluLayer   
    fullyConnectedLayer(16)
    reluLayer   
    fullyConnectedLayer(8)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer ];

lgraph = layerGraph(layers);
figure
plot(lgraph)

options = trainingOptions('sgdm', ...
    'MaxEpochs',1000,...
    'InitialLearnRate',1e-4, ...
    'Verbose',true, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',datastoreTest);

%% 8. Melatih Model
[net, validationInfo] = trainNetwork(datastoreTrain,layers,options);
analyzeNetwork(net)
%numel(net.Layers(end).ClassNames)

%% 9. Melakukan Prediksi atau Uji Kelas Gambar
imdsTest_rsz = augmentedImageDatastore(imageSize,imdsTest,'DataAugmentation',augmenter);
YPred = classify(net,imdsTest_rsz);
YTest = imdsTest.Labels;
%testaccuracy = sum(YPred == YTest)/numel(YTest);

figure;
idx = randperm(length(imdsTest_rsz.Files),64);
for i = 1:64
    subplot(8,8,i);
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

%% 10. Evaluasi Performansi Model Pengujian
%[YPredVal,scoresVal] = classify(net,datastoreTest);
%YVal = imdsVal.Labels;

% Perhitungan confussion matrix (accuracy, precision, recall, F1-score)
confusion_matrix = confusionmat(YTest,YPred);

classLabels = {'Biduran', 'Herpes', 'Kanker', 'Psoriasis'};
% Menampilkan hasil conf. matriks, akurasi, presisi, 
figure;
confusionchart(confusion_matrix,classLabels, ...
    "GridVisible","off", ...
    "OffDiagonalColor","#EDB120", ...
    "DiagonalColor","#D95319", ...
    "FontSize",9);
% Menambahkan Judul pada Plot
title('Confusion Matrix Classifikasi Penyakit Kulit');

% Hitung total prediksi dan aktual positif
total_positives = sum(sum(confusion_matrix));
actual_positives = sum(confusion_matrix(:, 4)); %ekstraks kolom keempat
predicted_positives = sum(confusion_matrix(4, :)); %ekstrakks baris keempat

% Hitung akurasi
accuracy = sum(diag(confusion_matrix)) / total_positives;

% Hitung presisi untuk setiap kelas (kolom)
precision = zeros(1, 4);
for i = 1:4
    precision(i) = confusion_matrix(i, i) / sum(confusion_matrix(:, i));
end

% Hitung recall untuk setiap kelas (baris)
recall = zeros(1, 4);
for i = 1:4
    recall(i) = confusion_matrix(i, i) / sum(confusion_matrix(i, :));
end

% Hitung F1-score untuk setiap kelas
f1_score = zeros(1, 4);
for i = 1:4
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Hitung rata-rata presisi dan recall
average_precision = mean(precision);
average_recall = mean(recall);

% Hitung F1-score rata-rata
f1_score_average = 2 * (average_precision * average_recall) / (average_precision + average_recall);

% Tampilkan hasil
disp('Akurasi:');
disp(accuracy);

disp('Presisi:');
disp(precision);

disp('Recall:');
disp(recall);

disp('F1-score:');
disp(f1_score);

disp('Rata-rata Presisi:');
disp(average_precision);

disp('Rata-rata Recall:');
disp(average_recall);

disp('Rata-rata F1-score:');
disp(f1_score_average);

%% 11. Menyimpan Hasil Model Klasifikasi
save net

%% 12. Uji Coba Prediksi Pada Gambar Baru
figure;
I = imread(".\PenyakitKulit\Psoriasis\Screenshot 2024-06-05 014703.jpg");
I2= imresize(I,[64,64],'nearest');
[Pred,scores] = classify(net,I2);
scores = max(double(scores*100));
imshow(I);
title(join([string(Pred),'' ,scores ,'%']))