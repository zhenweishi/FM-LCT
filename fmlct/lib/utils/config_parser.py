import argparse
from omegaconf import OmegaConf
import yaml
import monai
import sys
import os
from easydict import EasyDict as edict
from pathlib import Path
import shutil


__all__ = ["get_monai_conf", "get_omega_conf", "get_conf"]

def get_unique_name(run_name="", insert_front=True):
    import socket
    import datetime
    hostname = socket.gethostname()
    date_str = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{hostname}"

    if insert_front:
        return f"{(run_name + '_') if run_name else ''}{date_str}"
    else:
        return f"{date_str}{('_' + run_name) if run_name else ''}"

def get_monai_conf(conf_file):
    """
    https://docs.monai.io/en/latest/config_syntax.html#monai-bundle-configuration
    """
    # First of all, parse config file
    conf_file = Path(conf_file)
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"

    parser = monai.bundle.ConfigParser(yaml.load(open(conf_file), Loader=yaml.FullLoader))
    conf = parser.get_parsed_content()
    conf = edict(conf)
    return conf

def get_omega_conf(conf_file):
    # First of all, parse config file
    conf_file = Path(conf_file)
    assert os.path.exists(conf_file), f"Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)
    ret = edict()
    for key in conf:
        ret[key] = conf[key]
    return ret

def get_conf(conf_file, conf_parser="monai"):
    conf_file = Path(conf_file)
    assert conf_parser in ['monai', 'omega', 'auto'], f"Config parser {conf_parser} not supported!"

    if conf_parser == 'monai':
        conf = get_monai_conf(conf_file)
    elif conf_parser == 'omega':
        conf = get_omega_conf(conf_file)
    elif conf_parser == 'auto':
        print("Use auto config parser may cause unexpected errors!")
        conf = get_omega_conf(conf_file)
        conf_str = yaml.dump(dict(**conf))
        parser = monai.bundle.ConfigParser(yaml.load(conf_str, Loader=yaml.FullLoader))
        conf = parser.get_parsed_content()
        conf = edict(conf)
    else:
        raise NotImplementedError
    
    # run_name = get_unique_name(getattr(conf, "run_name", ""))
    run_name = getattr(conf, "run_name", "")
    output_root = getattr(conf, "output_root", "runs")
    conf.ckpt_dir = os.path.join(output_root, run_name, "ckpts")
    conf.log_dir = os.path.join(output_root, run_name, "logs", get_unique_name())

    cfg_bak_path = Path(output_root) / run_name / "cfgs" / get_unique_name(conf_file.name, insert_front=False)
    cfg_bak_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(conf, "save_cfg", False):
        # conf_file.link_to(cfg_bak_path) 
        shutil.copy(conf_file, cfg_bak_path)
    return conf
