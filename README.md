# ConQueR (SIGIR'23)

This is the official repository for our paper "ConQueR: Contextualized Query Reduction using Search Logs" in SIGIR'23.

## Dataset
Unfortunately, the search log dataset is not publicly available due to privacy issues. Instead, we provide sample data as an example of the data format.

### Install python environment

```bash
conda env create -n qr python==3.7.7 
```

### Activate environment
```bash
conda activate qr
pip install -r requirements.txt
```

---

## Reproducibility
### Usage

#### In terminal
- Run the python file (at the root of the project)
```bash
# run conquer_core
python conquer_core.py

```

```bash
# run conquer_sub
python conquer_sub.py
```
- Run the python file (at the root of the project) with best_ckpt file of ConQueR_core and ConQueR_sub in saves folder.
```bash
# run conquer_sub
python eval_ensemble.py
```

#### Arguments (see more arguments in `config.py`)
- gpus
    - default: 0


