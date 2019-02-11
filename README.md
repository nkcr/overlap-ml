# Overlapping with language modelling

Hold experiments on 3 models using the overlapping:

- **simple-lstn**, a very basic lstm
- **awd-lstm**, [AWD](https://arxiv.org/abs/1708.02182) ASGD Weight-Dropped LSTM
- **mos-lstm**, [MOS](https://arxiv.org/abs/1711.03953) Mixture of Softmaxes

To specify which model to run, use `--main-model {simple-lstm | awd-lstm | mos-lstm}` argument. There are additional common paramaters, as well as specific parameters for each model. Those can be found in `main_run.py`.

## Set-up

Download the data (PTB, WT2, WT103):

```bash
chmod +x get_data.sh
./get_data.sh
```

For emotions, add in `data/IEMOCAP/` the `all_features_cv` files.

Install dependencies:

```bash
python3.6 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## About the structure

Each model is in its corresponding folder. The `main_run.py` is responsible of handling the params and the common stuff among the models. 
It then loads the corresponding model, which means importing the right `main.py` file.

## Simple lstm

Basic run:

```bash
python3 main_run.py --main-model simple-lstm --epochs 1000
```

## MOS lstm

Basic run:

```bash
python3 main_run.py --main-model mos-lstm --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch-size 12 --lr 20.0 --epochs 1000 --nhid 960 --nhidlast 620 --emsize 280 --n-experts 15
```

## AWD lstm

Basic run:

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epochs 500
```

## Emotions (WIP)

Simple run with no order:

```bash
python3 main_run.py --main-model emotions-simple-lstm --order complete_random
```

# Commands to reproduce the experiments

## AWD PTB

**No order**:

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --seed-shuffle 141 --epoch 1000 --shuffle-full-seq
```

**Local order**:

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --seed-shuffle 141 --epoch 1000 --shuffle-row-seq
```

**Standard order**:

```bash
python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1000
```

**Overlapping {2,5,7,10}**:

```bash
overlaps=(2 5 7 10)
epochs=1000
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --batch-size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

## AWD WT2

**No order**

```bash
python3 main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --shuffle-full-seq
```

**Local order**

```bash
python main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --shuffle-row-seq
```

**Standard order**

```bash
python3 main_run.py --main-model awd-lstm --epochs 750 --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80
```

**Overlapping {2,5,7,10}**

```bash
overlaps=(2 5 7 10)
epochs=750
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model awd-lstm --data /data/noemien.kocher/datasets/wikitext-2 --dropouth 0.2 --seed 1882 --batch-size 80 --epochs "$(($epochs/$k))" --init-seq "overlapCN_${k}"
    sleep 10
done
```

## AWD WT103

**No order**

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN --shuffle-full-seq
```

**Local order**

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN --shuffle-row-seq
```

**Standard order**

```bash
python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when 12 --model QRNN
```

**Overlapping {2,5,7,10}**

```bash
# base num epochs is 14
overlaps=(2 5 7 10)
when_steps=147456
max_steps=172032
for i in "${!overlaps[@]}"
do
        :
        python3 -u main_run.py --main-model awd-lstm --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch-size 60 --optimizer adam --lr 1e-3 --data /data/noemien.kocher/datasets/wikitext-103 --when-steps "$when_steps" --model QRNN --init-seq "overlapCN_${overlaps[$i]}" --log-dir /data/noemien.kocher/logs/ --max-steps "$max_steps"
        sleep 10
done
```

## Simple PTB

**No order**:

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --shuffle-full-seq
```

**Local order**:

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --shuffle-row-seq
```

**Standard order**:

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1
```

**Overlapping {2,5,7,10}**:
```bash
overlaps=(2 5 7 10)
epochs=100
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model simple-lstm --epochs "$(($epochs/$k))" --batch-size 20 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1500 --lr-decay 1 --init-seq "overlapCN_${k}"
    sleep 10
done
```

## Simple WT2

**No order**

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --shuffle-full-seq
```

**Local order**

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --shuffle-row-seq
```

**Standard order**

```bash
python3 main_run.py --main-model simple-lstm --epochs 100 --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2
```

**Overlapping {2,5,7,10}**

```bash
overlaps=(2 5 7 10)
epochs=100
for k in "${overlaps[@]}"
do
    :
    python3 main_run.py --main-model simple-lstm --epochs "$(($epochs/$k))" --batch-size 80 --dropout 0.15 --nlayers 2 --bptt 70 --nhid 1150 --lr-decay 1 --data /data/noemien.kocher/datasets/wikitext-2 --init-seq "overlapCN_${k}"
    sleep 10
done
```

## MOS PTB

**Standard order**:

```bash
python main.py --main-model mos-lstm --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15
```

**Overlapping {2,5,7,10}**:

```bash
python main.py --main-model mos-lstm --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --init-seq overlapCN_7
```


# Acknowledgements

Code is heavily borrowed from the following sources:

- simple-lstm: https://github.com/deeplearningathome/pytorch-language-model
- awd-lstm: https://github.com/salesforce/awd-lstm-lm
- mos-lstm: https://github.com/zihangdai/mos