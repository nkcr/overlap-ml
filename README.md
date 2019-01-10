# Overlap with language modelling

Hold experiments on 2 models using overlap:

- **simple-lstn**, a very basic lstm
- **mos-lstm**, [MOS](https://arxiv.org/abs/1711.03953) Mixture of Softmaxes

To specify which model to run, use `--main-model {simple-lstm | mos-lstm}`. There are additional common paramaters as well as specific parameters for each model. Those can be found in `main_run.py`.

## Simple lstm

```bash
python3 main_run.py --main-model simple-lstm --epochs 1000
```

## MOS lstm

TODO