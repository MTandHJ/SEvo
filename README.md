

# Graph-enhanced Optimizers for <u>S</u>tructure-aware Recommendation Embedding <u>Evo</u>lution


![](pic/2023-09-22-13-54-03.png)


## Requirements

Python==3.9 | [PyTorch==1.12.1](https://pytorch.org/) | [freerec==0.4.3](https://github.com/MTandHJ/freerec)

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

