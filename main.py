import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils 
from seq_scripts import seq_train, seq_eval
from torch.cuda.amp import autocast as autocast
from utils.misc import *

def _unwrap(model: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model

def compute_gflops_with_fvcore(model: nn.Module, vid, vid_lgt):
    """
    Returns (gflops_total_batch, gflops_per_sample) or (None, None) if fvcore not available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        return None, None

    m = _unwrap(model).eval()
    with torch.inference_mode():
        # FLOPs for a *single forward* with current shapes
        flops_total = FlopCountAnalysis(m, (vid, vid_lgt)).total()
    gflops_total = flops_total / 1e9
    gflops_per_sample = gflops_total / max(1, int(vid.shape[0]))
    return gflops_total, gflops_per_sample



class Processor():
    def __init__(self, arg):
        self.arg = arg
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        shutil.copy2('./modules/tconv.py', self.arg.work_dir)
        shutil.copy2('./modules/resnet.py', self.arg.work_dir)
        shutil.copy2('./modules/gcn_lib/temgraph.py', self.arg.work_dir)
        if getattr(self.arg, "ablation_cfg", None):
            try:
                shutil.copy2(self.arg.ablation_cfg, self.arg.work_dir)
            except Exception:
                pass
        torch.backends.cudnn.benchmark = True
        if type(self.arg.device) is not int:
            init_distributed_mode(self.arg)
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()


        # self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

        # print(f"==================== PRINTING MODEL PROPERTIES ====================")
        print(f"{self.model}\n ==============================================================")
        total, trainable = self.count_params()
        print(f"Total parameters (total, trainable): {total}, {trainable}")
        # NEW: profile FLOPs once
        self.profile_gflops_once()
        # print(f"==================== PRINTING MODEL ====================")


    def start(self):
        if self.arg.phase == 'train':
            best_dev = {"wer":200.0, "del":100.0,"ins":100.0}
            best_tes = {"wer": 200.0, "del": 100.0, "ins": 100.0}
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            if is_main_process():
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.optimizer_args['num_epoch']):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                dev_wer={}
                dev_wer['wer']=0
                if is_main_process():
                    if eval_model:
                        dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                           'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                           'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        self.recoder.print_log("Dev WER: {:05.2f}% DEV del {:05.2f}% DEV ins {:05.2f}%".format(dev_wer['wer'], dev_wer['del'], dev_wer['ins']))
                        self.recoder.print_log("Test WER: {:05.2f}% Test del {:05.2f}% Test ins {:05.2f}%".format(test_wer['wer'], test_wer['del'],
                                                                                            test_wer['ins']))

                    if dev_wer["wer"] < best_dev["wer"]:
                        best_dev = dev_wer
                        best_tes = test_wer
                        best_epoch = epoch
                        model_path = os.path.join(self.arg.work_dir, "_best_model.pt")
                        self.save_model(epoch, model_path)
                        self.recoder.print_log('Save best model')

                    self.recoder.print_log('Best_dev: {:05.2f}, {:05.2f}, {:05.2f}, '
                                           'Best_test: {:05.2f}, {:05.2f}, {:05.2f},'
                                           'Epoch : {}'.format(best_dev["wer"], best_dev["del"], best_dev["ins"],
                                                               best_tes["wer"],best_tes["del"],best_tes["ins"], best_epoch))
                    if save_model:
                        # model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer['wer'], epoch)
                        fname = f"dev_{dev_wer['wer']:.2f}_epoch{epoch}_model.pt"
                        model_path = os.path.join(self.arg.work_dir, fname)
                        seq_model_list.append(model_path)
                        print("seq_model_list", seq_model_list)
                        self.save_model(epoch, model_path)
                    epoch_time = time.time() - epoch_time
                    total_time += epoch_time
                    torch.cuda.empty_cache()
                    self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
                self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test' and is_main_process():
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
                                 "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )
    def count_params(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
    
    def profile_gflops_once(self):
        """
        Uses one dev batch to estimate GFLOPs.
        Works for all ablations because it forwards (vid, vid_lgt).
        """
        try:
            loader = self.data_loader.get("dev", None) or self.data_loader.get("train_eval", None)
            if loader is None:
                self.recoder.print_log("[FLOPs] No loader available; skipping.")
                return
            data = next(iter(loader))
            vid = self.device.data_to_device(data[0])
            vid_lgt = self.device.data_to_device(data[1])
            gtot, gper = compute_gflops_with_fvcore(self.model, vid, vid_lgt)
            if gtot is None:
                self.recoder.print_log("[FLOPs] fvcore not installed. Run: pip install fvcore")
                return
            self.recoder.print_log(f"FLOPs: {gtot:.2f} GFLOPs per batch (~{gper:.2f} per sample).")
        except StopIteration:
            self.recoder.print_log("[FLOPs] Empty loader; skipping.")
        except Exception as e:
            self.recoder.print_log(f"[FLOPs] Skipped due to: {repr(e)}")



    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        if len(self.device.gpu_list) > 1:
            model = self.model.module
        else:
            model = self.model

        payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
        }
        if hasattr(self, "rng"):             # only if random_fix was enabled
            payload['rng_state'] = self.rng.save_rng_state()

        torch.save(payload, save_path)


    def loading(self):
        self.device.set_device(self.arg.device)

        # inside Processor.loading(), right after self.device.set_device(self.arg.device)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        logical = self.device.output_device  # e.g., 0 after remap
        print("CUDA_VISIBLE_DEVICES:", visible)
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("Using logical cuda:", logical)
        print("GPU selected name:", torch.cuda.get_device_name(logical))

        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        self.kernel_sizes = model.conv1d.kernel_size
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            print("using dataparalleling...")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model.to(self.arg.local_rank))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.arg.local_rank])
        else:
            model.cuda()
        # model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        s_dict = model.state_dict()
        for name in weights:
            if name not in s_dict:
                print(name)
                continue
            if s_dict[name].shape == weights[name].shape:
                s_dict[name] = weights[name]
        model.load_state_dict(s_dict, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.arg.local_rank)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log(f"Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading Dataprocessing")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading Dataprocessing finished.")
    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)


    # def build_dataloader(self, dataset, mode, train_flag):
    #     if len(self.device.gpu_list) > 1:
    #         if train_flag:
    #             sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=train_flag)
    #         else:
    #             sampler = torch.utils.data.SequentialSampler(dataset)
    #         batch_size = self.arg.batch_size if mode == "train" else self.arg.test_batch_size
    #         loader = torch.utils.data.DataLoader(
    #             dataset,
    #             sampler=sampler,
    #             batch_size=batch_size,
    #             collate_fn=self.feeder.collate_fn,
    #             num_workers=self.arg.num_worker,
    #             pin_memory=True,
    #             worker_init_fn=self.init_fn,
    #         )
    #         return loader
    #     else:
    #         return torch.utils.data.DataLoader(
    #             dataset,
    #             batch_size= self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
    #             shuffle=train_flag,
    #             drop_last=train_flag,
    #             num_workers=self.arg.num_worker if train_flag else 0,  # if train_flag else 0
    #             collate_fn=self.feeder.collate_fn,
    #             pin_memory=True,
    #             prefetch_factor= 4 if train_flag else 2,
    #             persistent_workers= train_flag if self.arg.num_worker > 0 else False,
    #             worker_init_fn=self.init_fn if self.arg.num_worker > 0 else None,
    #         )

    def build_dataloader(self, dataset, mode, train_flag):
        batch_size = self.arg.batch_size if mode == "train" else self.arg.test_batch_size

        if len(self.device.gpu_list) > 1:
            # DDP branch
            if train_flag:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

            nw = int(self.arg.num_worker) if train_flag else 0
            kwargs = dict(
                dataset=dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.feeder.collate_fn,
                num_workers=nw,
                pin_memory=True,
            )
            if nw > 0:
                kwargs.update(
                    dict(
                        prefetch_factor=4 if train_flag else 2,
                        persistent_workers=True,
                        worker_init_fn=self.init_fn,
                    )
                )
            return torch.utils.data.DataLoader(**kwargs)

        else:
            # single-GPU branch
            nw = int(self.arg.num_worker) if train_flag else 0
            kwargs = dict(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=nw,
                pin_memory=True,
                collate_fn=self.feeder.collate_fn,
            )
            if nw > 0:
                kwargs.update(
                    dict(
                        prefetch_factor=4 if train_flag else 2,
                        persistent_workers=True,
                        worker_init_fn=self.init_fn,
                    )
                )
            return torch.utils.data.DataLoader(**kwargs)



def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    # sparser = utils.get_parser()
    # p = sparser.parse_args()
    # if p.config is not None:
    #     with open(p.config, 'r') as f:
    #         try:
    #             default_arg = yaml.load(f, Loader=yaml.FullLoader)
    #         except AttributeError:
    #             default_arg = yaml.load(f)
    #     key = vars(p).keys()
    #     for k in default_arg.keys():
    #         if k not in key:
    #             print('WRONG ARG: {}'.format(k))
    #             assert (k in key)
    #     sparser.set_defaults(**default_arg)
    # args = sparser.parse_args()
    # with open(f"./configs/{args.dataset}.yaml", 'r') as f:
    #     args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    # processor = Processor(args)
    # utils.pack_code("./", args.work_dir)
    # processor.start()
 
    from argparse import Namespace
    from utils.parameters import load_yaml, merge_cfgs, get_parser

    # 1) Build parser (must include --config and --ablation_cfg)
    parser = get_parser()
    # First parse: just to read which files to load
    cmd = parser.parse_args()

    # 2) Seed defaults with baseline.yaml if provided
    if cmd.config is not None:
        base_cfg = load_yaml(cmd.config)  # dict from baseline.yaml
        parser.set_defaults(**base_cfg)
    else:
        base_cfg = {}

    # 3) Merge ablation.yaml on top (if provided), then set as defaults
    #    This ensures ablation overrides baseline, but CLI still wins.
    if getattr(cmd, "ablation_cfg", None):
        merged = merge_cfgs(base_cfg, cmd.ablation_cfg)  # dict
        parser.set_defaults(**merged)

    # 4) Re-parse with new defaults so CLI flags remain highest precedence
    args = parser.parse_args()

    # 5) Load dataset_info (unchanged)
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # 6) Start processor
    processor = Processor(args)

    # 7) Optional: archive configs for reproducibility
    # utils.pack_code("./", args.work_dir)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        utils.pack_code("./", args.work_dir)
    with open(os.path.join(args.work_dir, "code_snapshot.log"), "w") as f:
        f.write(buf.getvalue())

    # copy the *actual* config files used
    if cmd.config:
        shutil.copy2(cmd.config, os.path.join(args.work_dir, os.path.basename(cmd.config)))
    if getattr(cmd, "ablation_cfg", None):
        shutil.copy2(cmd.ablation_cfg, os.path.join(args.work_dir, os.path.basename(cmd.ablation_cfg)))

    processor.start()
