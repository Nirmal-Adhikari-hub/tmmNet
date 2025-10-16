import pdb
import time, os


class Recorder(object):
    def __init__(self, work_dir, print_log, log_interval):
        self.cur_time = time.time()
        self.print_log_flag = print_log
        self.log_interval = log_interval
        self.log_path = '{}/log.txt'.format(work_dir)
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

        # --- NEW: optional wandb ---
        self.wandb = None
        try:
            proj = os.getenv("WANDB_PROJECT", 'tmmNet')
            if proj:
                import wandb
                run_name = os.getenv("WANDB_RUN_NAME", 'deblurred_baseline_resnet34')
                wandb_dir = work_dir.rstrip("/")
                wandb.init(project=proj, name=run_name, dir=wandb_dir)
                self.wandb = wandb
                self.print_log(f"[wandb] initialized project={proj} name={run_name}", print_time=False)
        except Exception as e:
            self.wandb = None
            self.print_log(f"[wandb] disabled ({e})", print_time=False)
        # --- end wandb ---

        if self.wandb is not None:
            # make epochs the x-axis for eval metrics
            self.wandb.define_metric("epoch")
            self.wandb.define_metric("dev/*", step_metric="epoch")
            self.wandb.define_metric("test/*", step_metric="epoch")
            self.wandb.define_metric("train/*", step_metric="step")
            self.wandb.define_metric("train_epoch/*", step_metric="epoch")

    def log_metrics(self, metrics: dict, step=None):
        if self.wandb is not None:
            try:
                if step is not None:
                    self.wandb.log(metrics, step=step)
                else:
                    self.wandb.log(metrics)
            except Exception as _:
                pass

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, path=None, print_time=True):
        if path is None:
            path = self.log_path
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        try:
            from tqdm import tqdm
            tqdm.write(str)
        except Exception:
            print(str)        
        if self.print_log_flag:
            with open(path, 'a') as f:
                f.writelines(str)
                f.writelines("\n")

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def timer_reset(self):
        self.cur_time = time.time()
        self.timer = dict(dataloader=0.001, device=0.001, forward=0.001, backward=0.001)

    def record_timer(self, key):
        self.timer[key] += self.split_time()

    def print_time_statistics(self):
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(self.timer.values()))))
            for k, v in self.timer.items()}
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [GPU]{device}, [Forward]{forward}, [Backward]{backward}'.format(
                **proportion))
