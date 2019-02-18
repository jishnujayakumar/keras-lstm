# keras-lstm
This is a sample LSTM implementation in Keras

#### Setup
```
pip install -r requirements.txt
mkdir -p logs models
```
#### Train 
```
python train.py --lr 1e-3 --decay 1e-5 --vs 0.2 --bs 10 --epochs 100 --shuffle True
```

#### Help 
```
python train.py -h
```
	
