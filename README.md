# keras-lstm
This is a sample LSTM implementation in Keras

#### Setup
```
git clone https://github.com/jishnujayakumar/keras-lstm
chmod -R 777 keras-lstm
cd keras-lstm
pip install -r requirements.txt
mkdir -p logs models
```
#### Train 
```
python train.py --lr 1e-3 --decay 1e-5 --vs 0.2 --bs 128 --epochs 100 --shuffle True
```

#### To use Tensorboard 
```
tensorboard --logdir logs/
```

#### Help 
```
python train.py -h
```
	
