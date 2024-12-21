<div align="center">
  <h1>
    <b>
      Cross-modal Contrastive Learning for Unified Placenta Analysis Using Photographs
    </b>
  </h1>
  <h4>
    <b>
      <a href="https://doi.org/10.1016/j.patter.2024.101097" target="_blank">[Cell Press Patterns]</a><br>
      Yimu Pan, Manas Mehta, Jeffery A. Goldstein, Joseph Ngonzi, Lisa M. Bebell, Drucilla J. Roberts, 
      Chrystalle Katte Carreon, Kelly Gallagher, Rachel E. Walker, Alison D. Gernand, and James Z. Wang
    </b>
  </h4>
</div>


## Introduction
This repository contains the training and evaluation code for PlacentaCLIP


## Dependencies
See the requirements.txt file.

## Datasets
Please contact James Z. Wang (jwang@ist.psu.edu).

## Experiments

### PlacentaCLIP
```
python train.py --exp_name='PlacentaCLIP+' --pooling_type='transformer' --num_epochs=30 --text_sample_method=boot_group --filter_threshold=0.9 --save_every=10 --reg_weight=0.5 --alpha=0.5 
python test.py --exp_name='PlacentaCLIP+' --pooling_type='transformer' --reg_weight=0.5 --alpha=0.5
```
### PlacentaCLIP+
```
python train.py --exp_name='PlacentaCLIP+' --pooling_type='transformer' --num_epochs=30 --text_sample_method=boot_group --filter_threshold=0.9 --save_every=10 --reg_weight=0.5 --alpha=0.5 --additional_data=true
python test.py --exp_name='PlacentaCLIP+' --pooling_type='transformer' --reg_weight=0.5 --alpha=0.5
```


### Robustness Evaluation
```
python test_robustness.py --exp_name='PlacentaCLIP' --pooling_type='transformer'
```

### External Validation
```
python test_must.py --exp_name='PlacentaCLIP+' --pooling_type='transformer'
```

## Acknowledgement

This repo uses code from [x-pool](https://github.com/layer6ai-labs/xpool), [OpenCLIP](https://github.com/mlfoundations/open_clip) and [WBAug](https://github.com/mahmoudnafifi/WB_color_augmenter/tree/master).


## Citation
```
@article{pan2024cross,
    title = {Cross-modal contrastive learning for unified placenta analysis using photographs},
    journal = {Patterns},
    volume = {5},
    number = {12},
    pages = {101097},
    year = {2024},
    issn = {2666-3899},
    doi = {https://doi.org/10.1016/j.patter.2024.101097},
    author = {Yimu Pan and Manas Mehta and Jeffery A. Goldstein and Joseph Ngonzi and Lisa M. Bebell and Drucilla J. Roberts and Chrystalle Katte Carreon and Kelly Gallagher and Rachel E. Walker and Alison D. Gernand and James Z. Wang}
}
```
