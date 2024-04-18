"""
$ cd universal_manipulation_interface
$ export PYTHONPATH="${PWD}:${PYTHONPATH}"
$ python tests/test_inference_speed.py


On 56-core machine, NVIDIA GeForce RTX 3090:

# warmup
dt: 3.1060
# average
dt: 0.1531

"""

import os
import subprocess
import time

import dill
import hydra
import torch
import tqdm
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def main():
    ckpt_path = os.path.expanduser("~/tmp/2024-04-timing/umi/cup_test.ckpt")
    obs_path = os.path.expanduser("~/tmp/2024-04-timing/umi/obs_list.pkl")

    with open(obs_path, "rb") as f:
        obs_dict_list = dill.load(f)

    # Transfer back to cpu
    obs_dict_list = [
        dict_apply(obs_dict, lambda x: x.cpu())
        for obs_dict in obs_dict_list
    ]

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    OmegaConf.save(cfg, "/tmp/dp_config.yaml")
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations

    device = torch.device('cuda')
    policy.eval().to(device)

    subprocess.run("./scripts/machine_info.sh")

    for obs_dict in obs_dict_list:
        t_start = time.perf_counter()
        obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
        result = policy.predict_action(obs_dict)
        action = result['action_pred'][0].detach().to('cpu').numpy()
        dt = time.perf_counter() - t_start
        print(f"dt: {dt:.4f}")


if __name__ == "__main__":
    main()
