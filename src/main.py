import json
import os
import argparse
from argparse import BooleanOptionalAction as StoreTrue
from utils.dotdic import DotDic
import warnings

from trainer import train, train_n_trials
from utils.deployment import deploy
from utils.test_deployed import test_deployed


def read_conf(path: str):
    """This function reads a .json file and returns it as a DotDic.
    It also adds some arguments that are dependent on other arguments. Having
    this coded in this function removes redundancy in the config file, making it
    easier to write different configurations.
    """
    with open(path, "r") as f:
        conf = json.load(f)

    # add some redundantarguments related to video shapes
    conf["num_frames"] = int(conf["clip_duration"] * conf["transform_params"]["sampling_rate"])
    conf["input_blob_size"] = (1, 3, conf["num_frames"], conf["transform_params"]["side_size"], conf["transform_params"]["side_size"])

    # add the filenames in which checkpoints and deployed models will be saved:
    quant_sufix = "_q" if conf["quantize"] else ""
    qat_sufix = quant_sufix + "at" if conf["quantization_aware_training"] else ""
    crop_sufix = conf["train_annotation"].split("-")[1].split("/")[0]
    run_id = f"{conf['model']}_{conf['clip_duration']}s{conf['num_frames']}f{qat_sufix}_{crop_sufix}"  # this variable is the identifier for a run
    conf["run_id"] = run_id
    conf["deployed_mobile"] = os.path.join(conf['deployed_models'], run_id) + ".ptl"
    conf["deployed_server"] = os.path.join(conf['deployed_models'], run_id) + ".pt"

    conf["checkpoint_file"] = os.path.join(conf['ckpt_folder'], run_id) + ".ckpt"

    conf["trials_results_dest"] = os.path.join(conf["trials_result_dir"], run_id) + ".csv"

    conf["gpus"] = 1 if conf["use_cuda"] else 0

    print("read configuration:")
    print(conf)

    return DotDic(conf)


def main(args):
    run_config = read_conf(args.conf_file)

    if args.train:
        print("Training")
        train(run_config)

    if args.trials:
        print("Training some trials")
        train_n_trials(run_config)

    if args.deploy:
        print("Deploying")
        deploy(run_config)

    if args.test_deployed:
        print("Testing Deployed")
        test_deployed(run_config)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    # conf file should contain all relevant information for the full pipeline.
    parser.add_argument("-conf", "--conf-file", type=str, help="The the file in which the configuration of the run is stored.")

    # arguments to select what actions are going to be executed
    parser.add_argument("-train", type=bool, action=StoreTrue)
    parser.add_argument("-trials", type=bool, action=StoreTrue)
    parser.add_argument("-deploy", type=bool, action=StoreTrue)
    parser.add_argument("-test", "--test-deployed", type=bool, action=StoreTrue)

    args = parser.parse_args()

    main(args)
