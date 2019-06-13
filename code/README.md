# Baseline code

## Requirements and Installation

pytorch>=1.0

```
pip install -r requirements.txt
```

## preprocessing data
Download [metadata](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) for baseline method and put them into prepro_data folder.


```
python3 gen_data.py --in_path ../data --out_path prepro_data
```

## relation extration

training:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

testing:
```
CUDA_VISIBLE_DEVICES=6 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_test --input_theta 0.3601
```

## evidence extration

training:
```
CUDA_VISIBLE_DEVICES=6 python3 train_sp.py --model_name LSTM_SP  --save_name checkpoint_BiLSTMSP  --train_prefix dev_train --test_prefix dev_dev
```

testing:
```
CUDA_VISIBLE_DEVICES=0 python3 test_sp.py --model_name LSTM_SP --save_name checkpoint_BiLSTMSP --train_prefix dev_train --test_prefix dev_test --input_theta 0.4619
```

## evaluation

dev result can evaluated by 
```
python3 evalutaion result.josn ../data/dev.json
```

test result should be submit to Codalab.



