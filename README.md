# code to run

## local
`python3.10 main.py --nEpochs 30 --savePath ./model_saves/cnnimage_model.pth --useFullDataset`
`python3.10 eval.py --savePath ./model_saves/cnnimage_model.pth --useFullDataset`
`python3.10 predict.py --savePath ./model_saves/cnnimage_model.pth --imgPath ./input/seg_pred/seg_pred/591.jpg`

## Google Colab
`python3.10 main.py --nEpochs 200 --savePath ../gdrive/MyDrive/model_saves/cnnimage_200epoch_3layer.pth --useFullDataset`