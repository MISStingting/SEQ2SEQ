改写自对联训练demo,尝试用estimator实现
source_vocab词典: ocr和std2者 词典的合并

### 版本问题
ImportError: cannot import name 'monitoring' from 'tensorflow.python.eager'，
pip install tensorflow==1.13.1

如果需要使用低版本，例如1.10.0 不支持 tf.compat.v1

`        
 tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]`
`t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
c = tf.concat([t1, t2], -1)`
` c array([[[ 1,  2,  7,  4],
        [ 2,  3,  8,  4]],
       [[ 4,  4,  2, 10],
        [ 5,  3, 15, 11]]])>  ` ``  


`final_sequence_lengths: [12  9 19  7 10  6  7  7  5 11  7  7 22  7  7 11]
in_seq_len: [12  9 19  7 10  6  7  7  5 11  7  7 22  7  7 11]
out_seq_len:[17 12 25  7  7 12  4  7  7 12 12  7 25  9  7 10]   `

log_device_placement = True : 是否打印设备分配日志
allow_soft_placement = True : 如果你指定的设备不存在，允许TF自动分配设备
tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)

This is a project use seq2seq model to play couplets (对对联)。 This project is written with Tensorflow. 
You can try the demo at [https://ai.binwang.me/couplet](https://ai.binwang.me/couplet).

Pre-requirements
--------------

* Tensorflow>==1.13.1
* Python 3.6
* Dataset


Dataset
-----------

You will need some data to run this program, 
the dataset can be downloaded from [this project](https://github.com/wb14123/couplet-dataset).

** Note: If you are using your own dataset, you need to add '<blank>'and'`<s>` and `<\s>` as the first three line into the vocabs file. **

<blank>
<s>
</s>
------------

### Train

python -m seq2seq.runner --mode=train_and_eval --params_file=seq2seq/config.yml


### Run the trained model

Then run `python serving.py` will start a web service that can play couplet.


Examples
-------------
Here are some examples generated by this model:

| 上联                        | 下联                |
|-----------------------------|--------------------|
| 殷勤怕负三春意                | 潇洒难书一字愁        |
| 如此清秋何吝酒                | 这般明月不须钱        |
| 天朗气清风和畅                | 云蒸霞蔚日光辉        |
| 梦里不知身是客                | 醉时已觉酒为朋        |
| 千秋月色君长看                | 一夜风声我自怜        |

Result
-------------
取训练数据70万训练

