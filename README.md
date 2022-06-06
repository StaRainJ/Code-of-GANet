# Code-of-GANet
code for ""Learning for mismatch removal via graph attention networks

Authors: Xingyu Jiang, Yang Wang, Aoxiang Fan and Jiayi Ma


# Requirements
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.

# Run the code
### Test pretrained model

We provide the model trained on YFCC100M and SUN3D described in our paper. Run the test script to get results in our paper.
```bash
bash test.sh
```
### Test model on your own data-- see './Data'
```bash
python main4OwnData.py --model_path="./log/main.py/test/" 
```

### Test or train model on public  YFCC100M and SUN3D dataset
The HDF5 file is provide by the repo zjhthu/OANet, Please follow their way to generate the training/valid/testing set.
After generating dataset for YFCC100M/SUN3D, run the following 

### Test model on YFCC100M
```bash
python main.py --use_ransac=False --data_te='/data/yfcc-sift-2000-test.hdf5' --run_mode='test'
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

### Test model on SUN3D
```bash
python main.py --use_ransac=False --data_te='/data/sun3d-sift-2000-test.hdf5' --run_mode='test'
```




### Train model on YFCC100M



```bash
python main.py --run_mode= 'train'
```

You can train the fundamental estimation model by setting `--use_fundamental=True --geo_loss_margin=0.03` and use side information by setting `--use_ratio=2 --use_mutual=2`

