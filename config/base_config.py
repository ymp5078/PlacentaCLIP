from abc import abstractmethod, ABC


class Config(ABC):
    def __init__(self):
        args = self.parse_args()
        
        self.dataset_name = args.dataset_name
        self.data_dir = args.data_dir
        self.msrvtt_train_file = args.msrvtt_train_file
        self.num_frames = args.num_frames
        self.video_sample_type = args.video_sample_type
        self.input_res = args.input_res
        self.text_sample_method=args.text_sample_method

        self.exp_name = args.exp_name
        self.model_path = args.model_path 
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.log_step = args.log_step
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch
        self.eval_window_size = args.eval_window_size
        self.metric = args.metric

        self.huggingface = args.huggingface
        self.arch = args.arch
        self.clip_arch = args.clip_arch
        self.embed_dim = args.embed_dim

        self.loss = args.loss
        self.precision = args.precision
        self.clip_lr = args.clip_lr
        self.noclip_lr = args.noclip_lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion
    
        self.pooling_type = args.pooling_type
        self.k = args.k
        self.attention_temperature = args.attention_temperature
        self.num_mha_heads = args.num_mha_heads
        self.transformer_dropout = args.transformer_dropout
        self.filter_threshold = args.filter_threshold

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir
        self.checkpoint = args.checkpoint
        self.alpha = args.alpha
        self.reg_weight = args.reg_weight

        self.robustness_train_severity = args.robustness_train_severity
        self.additional_data = args.additional_data

        self.text_condition = args.text_condition

   
    @abstractmethod
    def parse_args(self):
        raise NotImplementedError

