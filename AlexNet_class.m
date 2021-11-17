DatasetPath='C:\Users\VIGNESH S\Desktop\ECG\ECG Signal Processing\ecgdataset';
images = imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames'); 
numTrainFiles = 250;
[TrainImages,TestImages] = splitEachLabel(images,numTrainFiles,'randomize');

net=alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses =3;
layers = [ 
    layersTransfer 
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer]; 
options = trainingOptions('sgdm', 'MiniBatchSize',20, 'MaxEpochs',8, 'InitialLearnRate',1e-4, 'Shuffle','every-epoch', 'ValidationData',TestImages, 'ValidationFrequency',10, 'Verbose',true, 'Plots','training-progress');
netTransfer = trainNetwork(TrainImages,layers,options);
YPred = classify(netTransfer,TestImages);
YValidation = TestImages.Labels; 
accuracy = sum(YPred==YValidation)/numel(YValidation)
plotconfusion(YValidation,YPred)