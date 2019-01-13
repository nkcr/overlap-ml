# Overlap with language modelling

Hold experiments on 2 models using overlap:

- **simple-lstn**, a very basic lstm
- **mos-lstm**, [MOS](https://arxiv.org/abs/1711.03953) Mixture of Softmaxes

To specify which model to run, use `--main-model {simple-lstm | mos-lstm}`. There are additional common paramaters as well as specific parameters for each model. Those can be found in `main_run.py`.

## Set up steps

**1) Download the data**:

```bash
chmod +x get_data.sh
./get_data.sh
```

***2) Install dependencies**:

```bash
python3.6 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Simple lstm

```bash
python3 main_run.py --main-model simple-lstm --epochs 1000
```

## MOS lstm

TODO

## AWD lstm

Fast run:

```bash
python3 main_run.py --batch-size 20 --data data/penn --dropouti 0.4 \
                    --dropouth 0.25 --seed 141 --epoch 1 \
                    --nhid 5 --emsize 5 --nlayers 1 --bptt 5 \
                    --main-model awd-lstm
```