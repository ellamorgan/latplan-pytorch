# Latplan-pytorch
Requires pytorch==1.9.1 (current version), wandb

To run:
```
python3 main.py --data puzzle
```

Options for data are:  
&emsp;puzzle  
(More to be added later)

Flags:  
&emsp;-no_cuda (train on CPU - will do this automatically if CUDA not available)  
&emsp;-no_wandb (don't log results with weights & biases)

Example with flags:
```
python3 main.py --data puzzle -no_wandb
```
