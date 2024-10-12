import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'VLP placenta'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='NU_Pretrain', help="Dataset name")
        parser.add_argument('--data_dir', type=str, default='/ocean/projects/iri180005p/ymp5078/VLPlacenta/VLPlacenta/data/NU_nlp_data', help="Location of data")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=tuple, default=(384,512))
        parser.add_argument('--text_sample_method', default='group', help="'group'/'fix_group'/'boot_group'")

        # aug
        parser.add_argument('--robustness_train_severity', type=int, default=0)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, default='debug', required=False, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=50, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=1, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='RN50', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=1024, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--precision', type=str, default='amp')
        parser.add_argument('--clip_lr', type=float, default=1e-5, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-4, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=5)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')
        parser.add_argument('--checkpoint', type=str, default=None)

        # false negative filter_threshold
        parser.add_argument('--filter_threshold', type=float, default=0.9, help='Threshold to consider as false negative')
        parser.add_argument('--alpha', type=float, default=0.5, help='weight for the local contrastive loss')
        parser.add_argument('--reg_weight', type=float, default=0.5, help='weight for the regularization contrastive loss')
        parser.add_argument('--additional_data', type=bool, default=False, help='whether to include the additional data')

        # testing setting
        
        parser.add_argument('--text_condition', type=bool, default=False, help='whether to condition on text')
        
        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
