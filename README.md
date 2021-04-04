# MAML-TF2.2

  * paper: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

## Requirements

``` pip install requirements.txt```

## Datasets
### Omniglot
1. Download Omniglot dataset and extract the contents of images_background.zip and images_evaluation.zip to ./datasets/omniglot/
```
git clone git@github.com:brendenlake/omniglot.git
```
```
datasets/omniglot
|-- images_background
|-- images_evaluation
```
2. Modify parameters in config.py, such as batchsize, lr ,update step, ...
  - train model -> please set cfg.mode = True
  - test model -> please set cfg.mode = False
4. Run the main python script
```
python main.py
```
