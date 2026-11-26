# python3 train.py --train-set ./data/train.data --val-set ./data/val.data --model-path ./checkpoints/model.ckpt
python3 export.py -i ../../model.ckpt -o ../../model.onnx
