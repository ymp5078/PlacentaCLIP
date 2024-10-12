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
from trainer.evaluation import LinearClassifier
from modules.optimization import AdamW

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

    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    config.dataset_name = 'NU_Finetune'
    valid_data_loaders  = [DataFactory.get_data_loader(config, split_type='train'),
                            DataFactory.get_data_loader(config, split_type='val')]
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
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    loss = LossFactory.get_loss(config)
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loaders=valid_data_loaders,
                      lr_scheduler=None,
                      grad_scaler=None,
                      writer=writer,
                      tokenizer=tokenizer)
    
    # checkpoint_path = os.path.join(config.output_dir,config.exp_name, 'checkpoint-epoch350.pth')
    checkpoint_path = 'model_best.pth'
    print(os.path.join(config.output_dir,config.exp_name, checkpoint_path))
    # checkpoint_path = 'checkpoint-epoch20.pth'
    if os.path.exists(os.path.join(config.output_dir,config.exp_name, checkpoint_path)):
        trainer.load_checkpoint(checkpoint_path)
        # trainer.load_checkpoint('checkpoint-epoch350.pth')
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        print("Checkpoint loaded")
    # trainer.eval()


    # checkpoint = torch.load(checkpoint_path)
    # state_dict = checkpoint['state_dict']
    
    # model.load_state_dict(state_dict)

    
    config.dataset_name = "NU_Finetune"


    output = ""
    for topic in ['mir','chorioamnionitis']:#['mir_placentanet','chorioamnionitis_placentanet','mir','chorioamnionitis']:
        val_mAP = []
        val_AUC = []
        for random_seed in [100,200,300,400,500]:
            test_data_loaders = [DataFactory.get_split_data_loader(config, topic.split('_')[0],random_seed, split_type='train+val'),
                            DataFactory.get_split_data_loader(config, topic,random_seed, split_type='ipad')]
            linear_clf = LinearClassifier(
                    model=model,
                    dataloaders=test_data_loaders,
                    tokenizer = tokenizer,
                    text_train=False
                    # args=args
                )
            val_res = linear_clf.eval()
            print(topic,val_res)
            val_mAP.append(val_res['mAP'])
            val_AUC.append(val_res['auc'])
        val_mAP = np.array(val_mAP)
        val_AUC = np.array(val_AUC)
        output+=f'{topic} mAP: {val_mAP.mean()},{val_mAP.var()}   AUC: {val_AUC.mean()},{val_AUC.var()}\n'
        print(f'{topic} mAP: {val_mAP.mean()},{val_mAP.var()}   AUC: {val_AUC.mean()},{val_AUC.var()}')
    print(output)

    output = ""
    for topic in ['sepsis','meconium','fir','mir','chorioamnionitis']:
        val_mAP = []
        val_AUC = []
        for random_seed in [100,200,300,400,500]:
            test_data_loaders = [DataFactory.get_split_data_loader(config, topic,random_seed, split_type='train+val'),
                            DataFactory.get_split_data_loader(config, topic,random_seed, split_type='test')]
            linear_clf = LinearClassifier(
                    model=model,
                    dataloaders=test_data_loaders,
                    tokenizer = tokenizer,
                    text_train=False
                    # args=args
                )
            val_res = linear_clf.eval()
            print(topic,val_res)
            val_mAP.append(val_res['mAP'])
            val_AUC.append(val_res['auc'])
        val_mAP = np.array(val_mAP)
        val_AUC = np.array(val_AUC)
        output+=f'{topic} mAP: {val_mAP.mean()},{val_mAP.var()}   AUC: {val_AUC.mean()},{val_AUC.var()}\n'
        print(f'{topic} mAP: {val_mAP.mean()},{val_mAP.var()}   AUC: {val_AUC.mean()},{val_AUC.var()}')
    print(output)


    


if __name__ == '__main__':
    main()

