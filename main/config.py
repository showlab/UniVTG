import os
import pdb
import time
import torch
import logging
import argparse
import importlib
from utils.basic_utils import mkdirp, remkdirp, \
    load_json, save_json, make_zipfile, dict_to_markdown

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        # * Running configs
        parser.add_argument("--dset_type", type=str, choices=["mr", "hl", "vs", "vlp"])    # moment retrieval, highlight detection, and video summarization
        parser.add_argument("--dset_name", type=str, choices=["qvhighlights", "charades", "anet", "tvsum", "youtube", "summe", "ego4d", "qfvs", "video2gif", "coin", "hacs", "vlp", "videocc", "tacos"])
        parser.add_argument("--domain_name", type=str, default=None)
        parser.add_argument("--model_id", type=str, default="moment_detr")
        parser.add_argument("--exp_id", type=str, default="debug", help="id of this run, required at training")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")

        # * DDP
        parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--data_ratio", type=float, default=1.0,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # * Training configs
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=400, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--lr_gamma", type=float, default=0.1, help="lr reduces the gamma times after the `drop' epoch")
        parser.add_argument("--lr_warmup", type=float, default=-1, help="linear warmup scheme")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")

        # ** Loss coefficients
        # *** boundary branch
        parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                            help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
        parser.add_argument('--b_loss_coef', default=10, type=float)    # boundary regression e.g., l1
        parser.add_argument('--g_loss_coef', default=1, type=float) # giou loss
        # *** foreground branch
        parser.add_argument('--eos_coef', default=0.1, type=float, help="relative classification weight of the no-object class")
        parser.add_argument('--f_loss_coef', default=4, type=float) # cls loss for foreground
        # *** saliency branch
        parser.add_argument("--s_loss_intra_coef", type=float, default=1., help="inter-video (frame-level) saliency loss e.g. momentdetr saliency loss")
        parser.add_argument("--s_loss_inter_coef", type=float, default=0., help="intra-video (sample-level) saliency loss,")

        # * Eval configs
        parser.add_argument("--main_metric", type=str, default="MR-full-mAP")
        parser.add_argument('--eval_mode', default=None, type=str,
                            help="how to integrate foreground and saliency for better prediction")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--eval_epoch", type=int, default=5,
                            help="number of epochs for once inference")
        parser.add_argument("--eval_init", action="store_true", help="evaluate model before training i.e. `epoch=-1'")
        parser.add_argument("--save_interval", type=int, default=50)

        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_dir", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # ** NMS configs
        parser.add_argument("--no_sort_results", action="store_true",
                            help="do not sort results, use this for moment query visualization")
        parser.add_argument("--max_before_nms", type=int, default=10)
        parser.add_argument("--max_after_nms", type=int, default=10)
        parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
        parser.add_argument("--nms_thd", type=float, default=-1,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")

        # * Dataset configs
        parser.add_argument("--use_cache",  type=int, default=-1, help="Preload features into cache for fast IO")
        parser.add_argument("--max_q_l", type=int, default=75)
        parser.add_argument("--max_v_l", type=int, default=75)
        parser.add_argument("--clip_length", type=float, default=1.0)
        parser.add_argument("--clip_len_list", type=int, nargs='+')
        parser.add_argument("--max_windows", type=int, default=5)

        parser.add_argument("--add_easy_negative", type=int, default=1)
        parser.add_argument("--easy_negative_only", type=int, default=1)
        parser.add_argument("--round_multiple", type=int, default=1)

        parser.add_argument("--train_path", type=str, default=None, nargs='+')
        parser.add_argument("--eval_path", type=str, default=None,
                            help="Evaluating during training, for Dev set. If None, will only do training, ")
        parser.add_argument("--train_path_list", type=str, nargs='+')
        parser.add_argument("--eval_path_list", type=str, nargs='+')
        parser.add_argument("--feat_root_list", type=str, nargs='+')

        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")
        parser.add_argument("--v_feat_types", type=str)
        parser.add_argument("--t_feat_type", type=str)

        # * Model configs
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to vid/txt projector")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")

        # ** Transformer
        parser.add_argument('--enc_layers', default=4, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--sub_enc_layers', default=2, type=int,
                            help="Number of encoding layers in the video / text transformer in albef-style.")
        parser.add_argument('--dec_layers', default=2, type=int,
                            help="Number of decoding layers in the transformer, N/A for UniVTG")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--droppath', default=0.1, type=float,
                            help="Droppath applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        # ** momentdetr configs e.g. Matcher, saliency margin
        parser.add_argument('--set_cost_span', default=10, type=float,
                            help="L1 span coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou span coefficient in the matching cost")
        parser.add_argument('--set_cost_class', default=4, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                            help="Disables auxiliary decoding losses (loss at each layer)")

        # * Query-Force Video Summarization
        parser.add_argument("--max_segment_num", type=int, default=20)
        parser.add_argument("--max_frame_num", type=int, default=200)
        parser.add_argument("--top_percent", type=float, default=0.02)

        parser.add_argument("--qfvs_vid_feature", type=str, default='fps1')
        parser.add_argument("--qfvs_txt_feature", type=str, default='query')
        parser.add_argument("--qfvs_split", type=int, default=-1)

        parser.add_argument("--qfvs_dense_shot", type=int, default=-1)
        parser.add_argument("--qfvs_score_ensemble", type=int, default=-1)
        parser.add_argument("--qfvs_score_gather", type=int, default=-1)
        parser.add_argument("--qfvs_loss_gather", type=int, default=-1)
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self, args=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        
        if args is not None:
            args_dict = vars(args)
            opt_dict = vars(opt)
            for key, value in args_dict.items():
                opt_dict[key] = value
            opt = argparse.Namespace(**opt_dict)    
            opt.model_dir = os.path.dirname(opt.resume)
            torch.cuda.set_device(opt.gpu_id)
            
        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",  "max_before_nms", "max_after_nms"
                               "max_pred_l", "min_pred_l", "gpu_id",
                               "resume", "resume_all", "no_sort_results",
                               "eval_path", "eval_split_name"]:
                            #    "dset_name", "v_feat_dirs", "t_feat_dir"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            # ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode

            if 'debug' not in opt.exp_id:
                opt.results_dir = os.path.join(opt.results_root, "-".join([opt.dset_type, opt.dset_name]), "-".join([opt.exp_id, opt.v_feat_types, opt.t_feat_type, time.strftime("%Y_%m_%d_%H")]))
            else:
                opt.results_dir = os.path.join(opt.results_root, "-".join([opt.dset_type, opt.dset_name]), opt.exp_id) # debug mode.

            if int(opt.local_rank) in [0, -1]:
                # mkdirp(opt.results_dir)
                remkdirp(opt.results_dir)   # remove dir and remkdir it.

                # save a copy of current code
                code_dir = os.path.dirname(os.path.realpath(__file__))
                code_zip_filename = os.path.join(opt.results_dir, "code.zip")
                make_zipfile(code_dir, code_zip_filename,
                            enclosing_dir="code",
                            exclude_dirs_substring="results",
                            exclude_dirs=["results", "debug_results", "__pycache__"],
                            exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        if int(opt.local_rank) in [0, -1]:
            self.display_save(opt)
            opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
            opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
            opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
            opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
            # opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")

        if int(opt.local_rank) in [-1]:
            torch.cuda.set_device(opt.gpu_id)
        opt.pin_memory = not opt.no_pin_memory

        if opt.local_rank == -1:
            torch.cuda.set_device(opt.gpu_id)

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt

class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")

class WarmupStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, step_size, gamma=self.gamma, last_epoch=last_epoch)
    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        # e.g. warmup_steps = 10, case: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21...
        if self.last_epoch == self.warmup_steps or(self.last_epoch % self.step_size != 0 and self.last_epoch > self.warmup_steps):
            return [group['lr'] for group in self.optimizer.param_groups]
        # e.g. warmup_steps = 10, case: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        elif self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * float(self.last_epoch + 1) / float(self.warmup_steps) for group in self.optimizer.param_groups]
        
        
        # e.g. warmup_steps = 10, case: 10, 20, 30, 40...
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * float(self.last_epoch) / (self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** ((self.last_epoch -  self.warmup_steps)// self.step_size) for base_lr in self.base_lrs]

def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")

    importer = importlib.import_module('.'.join(['model', opt.model_id]))
    model, criterion = importer.build_model(opt)

    if int(opt.device) >= 0:
        logger.info("CUDA enabled.")
        model.to(opt.gpu_id)
        criterion.to(opt.gpu_id)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)

    if opt.lr_warmup != -1 and opt.lr_drop > 0:
        lr_scheduler = WarmupStepLR(optimizer, warmup_steps=opt.lr_warmup[0], step_size=opt.lr_drop, gamma=opt.lr_gamma)
    
    elif opt.lr_warmup != -1:
        from transformers import get_constant_schedule_with_warmup
        lr_scheduler =  get_constant_schedule_with_warmup(optimizer, opt.lr_warmup[0])

    elif opt.lr_drop > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=opt.lr_gamma)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        
        for key in list(checkpoint["model"].keys()):
            checkpoint["model"][key.replace('module.', '')] = checkpoint["model"].pop(key)
        model.load_state_dict(checkpoint["model"])

        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler
