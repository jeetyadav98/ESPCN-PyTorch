# ESPCN

This repository is implementation of the ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/abs/1609.05158).


## Requirements
Install using ```pip3 install -r requirements.txt```

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/4mv1v4qfjo17zg3/91-image_x3.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/9qlb94in1iqh6nf/Set5_x3.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 200 \
                --num-workers 8 \
                --seed 123                
```

## Test

Pre-trained weights can be downloaded from the links below.

| Model | Scale | Link |
|-------|-------|------|
| ESPCN (91) | 3 | [Download](https://www.dropbox.com/s/2fl5jz5nw9oiw1f/espcn_x3.pth?dl=0) |

The results are stored in the same path as the query image.

```bash
python test.py --weights-file "BLAH_BLAH/espcn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```
