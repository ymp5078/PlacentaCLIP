import os
import torch
import random
import numpy as np
from config.vlp_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from torchvision import transforms

def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.huggingface:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    if config.robustness_train_severity > 0:
        train_data_loader = DataFactory.get_robustness_data_loader(config, split_type='train', severity=config.robustness_train_severity)
    else:
        train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    
    def plot(imgs, with_orig=False, row_title=None, name='aug_image.png', **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
                ax.imshow(np.asarray(invTrans(img).permute(1,2,0)), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        # plt.tight_layout()
        plt.savefig(f'dumps/{name}',bbox_inches='tight')
    config.dataset_name = 'NU_Finetune'
    valid_data_loaders  = [DataFactory.get_data_loader(config, split_type='train'),
                            DataFactory.get_data_loader(config, split_type='val')]
    # plot([train_data_loader.dataset[i]['image'] for i in range(5)])
    # plot([valid_data_loaders[0].dataset[i]['image'] for i in range(5)],name='val_img.png')
    model = ModelFactory.get_model(config)
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
      
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]

    # is_clip = lambda n, p: 'clip' in n
    # is_gain_or_bias_params = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    # # paramter groups
    # clip_gain_or_bias_params = [p for n, p in params_optimizer if is_gain_or_bias_params(n, p) and is_clip(n, p) and p.requires_grad]
    # clip_rest_params = [p for n, p in params_optimizer if not is_gain_or_bias_params(n, p) and is_clip(n, p) and p.requires_grad]
    # head_gain_or_bias_params = [p for n, p in params_optimizer if is_gain_or_bias_params(n, p) and not is_clip(n, p) and p.requires_grad]
    # head_rest_params = [p for n, p in params_optimizer if not is_gain_or_bias_params(n, p) and not is_clip(n, p) and p.requires_grad]
    # optimizer_grouped_params = [
    #     {'params': clip_gain_or_bias_params, 'lr': config.clip_lr},
    #     {'params': clip_rest_params, 'lr': config.clip_lr * 0.1},
    #     {'params': head_gain_or_bias_params, 'lr': config.noclip_lr},
    #     {'params': head_rest_params, 'lr': config.noclip_lr}
    # ]
    # print(len(clip_gain_or_bias_params),len(clip_rest_params),len(head_gain_or_bias_params),len(head_rest_params))
    # # print([n for n, p in params_optimizer if not is_gain_or_bias_params(n, p) and not is_clip(n, p) and p.requires_grad])
    # print("check params:",len(params_optimizer)==len(clip_gain_or_bias_params+clip_rest_params+head_gain_or_bias_params+head_rest_params))
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    # setup grad_scaler for AMP
    grad_scaler = torch.cuda.amp.GradScaler() if config.precision == 'amp' else None
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loaders=valid_data_loaders,
                      lr_scheduler=scheduler,
                      grad_scaler=grad_scaler,
                      writer=writer,
                      tokenizer=tokenizer,
                      alpha=config.alpha,
                      reg_weight=config.reg_weight)

    # traiin from checkpoint
    if config.checkpoint is not None and os.path.exists(config.checkpoint):
        # trainer.load_checkpoint('model_best.pth')
        checkpoint = torch.load(config.checkpoint)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print("Loading checkpoint: {} ...".format(config.checkpoint))
    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")  

    trainer.train()


if __name__ == '__main__':
    main()
