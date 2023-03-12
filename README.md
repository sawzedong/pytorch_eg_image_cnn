# code to run

## local
`python3.10 main.py --nEpochs 30 --savePath ./model_saves/cnnimage_model.pth --layers 3 --kernel_size 3`
`python3.10 eval.py --savePath ./model_saves/cnnimage_model.pth --layers 3 --kernel_size 3`
`python3.10 predict.py --savePath ./model_saves/cnnimage_model.pth --layers 3 --kernel_size 3 --imgPath ./input/seg_pred/seg_pred/591.jpg`

## Google Colab
`python3 main.py --nEpochs 50 --savePath ../gdrive/MyDrive/model_saves/cnnimage_50epoch_3layer_3k.pth --layers 3 --kernel_size 3`