

# Graph-enhanced Optimizers for <u>S</u>tructure-aware Recommendation Embedding <u>Evo</u>lution


![](pic/2023-09-22-13-54-03.png)


## Requirements

Use the following commands to prepare the environment (**CUDA: 11.3**):

```
conda create --name=PyT12 python=3.9; conda activate PyT12; bash setup_env.sh
```


## Usage

We provide configs and experimental logs for the Neumann series approximation with re-scaling. You can re-run them and try some other hyperparameters:

```
python main.py --config=configs/xxx.yaml --optimizer=AdamWSEvo --aggr=neumann --L=3 --beta3=0.99 --H=1
```

- optimizer: AdamWSEvo|AdamW|AdamSEvo|Adam|SGDSEvo|SGD
- aggr: neumann|average|momentum
- L: layers, int
- beta3: $\beta$
- H: Path length

