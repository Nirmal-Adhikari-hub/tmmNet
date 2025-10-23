import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)

        # Respect the shell's pinning if present
        vis_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not vis_env:
            # No pinning set by the shell → set it here once
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            vis_env = os.environ["CUDA_VISIBLE_DEVICES"]

        # Build logical gpu list from *visible* devices
        vis_list = [x for x in vis_env.split(",") if x != ""]
        self.gpu_list = list(range(len(vis_list)))
        self.output_device = self.gpu_list[0] if self.gpu_list else "cpu"

        # Pick the requested logical device if user passed "0,1" etc.
        # (we keep only the first as output device)
        if self.gpu_list:
            torch.cuda.set_device(self.output_device)

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device, non_blocking=True)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device, non_blocking=True)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device, non_blocking=True)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device, non_blocking=True)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        # if len(gpus) == 0:
        #     torch.zeros(1).cuda()
        # else:
        #     gpus = [gpus] if isinstance(gpus, int) else list(gpus)
        #     for g in gpus:
        #         torch.zeros(1).cuda(g)
        return
