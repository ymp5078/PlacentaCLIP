<div align="center">
<h1>
<b>
Cross-modal Contrastive Learning for Unified Placenta Analysis Using Photographs
</b>
</h1>
<h4>
<b>
Yimu Pan, Manas Mehta, Jeffery A. Goldstein, Joseph Ngonzi, Lisa M. Bebell, Drucilla J. Roberts, Chrystalle Katte Carreon, Kelly Gallagher, Rachel E. Walker,
Alison D. Gernand, and James Z. Wang  
</b>
</h4>
</div>

<a name="intro"/>

## Introduction
This repository contains the training and evaluation code for PlacentaCLIP

<a name="depend"/>

## Dependencies
See the requirements.txt file.

<a name="datasets"/>

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
