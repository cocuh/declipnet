declipnet
=========

音割れ音源を復元する深層学習モデル．

Deep neural network model for declipping.

C91 sigcoww L0 vol.1にて書いたものに近いがそのものではない．branchを見よ．

It uses WaveNets to predict conditional probability of declipped point from given past and future audio points.
In this branch, GAN technique is not used.
This code is a artifact of my hobby-research which is published as a doujinshi, C91 sigcoww L0 vol.1 pp.25-36.

https://sigcoww.github.io/posts/2016/11/06/c91.html

https://sigcoww.booth.pm/items/407104


使い方
------
`./data` に `*.ogg` ファイルをたくさん置き， `python main.py` を叩くと学習される．

`./result/log` にlogがdumpされるので， `tensorboard --logdir ./result/log`などするとlossなどが可視化される．

`./result/sample` に `.flac`でそのiteration時のサンプルが出力される．

`./result/snapshot` にモデルのsnapshotが作成されるのでテストデータで実験したい際はこれを利用する．

```console
$ mkdir data
$ cp ${何かの音源ファイルをたくさん}.ogg ./data
$ pip install numpy tensorflow-gpu soundfile librosa
$ python main.py
$ tensorboard --logdir ./result/log
```

special thanks
--------------
To impl this, I referenced the ibab's implementation.

https://github.com/ibab/tensorflow-wavenet
