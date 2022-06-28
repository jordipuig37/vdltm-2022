import os
import pandas as pd
import copy
import pytorch_lightning as pl

from utils.module import VideoClassifier
from utils.dataset import DataModule
from utils.callbacks import prepare_callbacks


def train(args):
    """This function aggregates all the modules needed to train a model. Using
    the checkpoint callback it saves the best model checkpoint.
    """
    my_callbacks = prepare_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args.as_namespace(), callbacks=list(my_callbacks.values()))

    module = VideoClassifier(**args)  # we pass the unpacked (**args) dictionary to be able to save each hyperparam
    data_module = DataModule(args)

    if hasattr(args, "train_from_ckpt"):
        trainer.fit(module, datamodule=data_module, ckpt_path=args.train_from_ckpt)
    else:
        trainer.fit(module, datamodule=data_module)

    # validate the best model
    trainer.validate(dataloaders=data_module)
    trainer.save_checkpoint(args.checkpoint_file)

    # on test start the module is converted to quantized form but then it's never saved
    trainer.test(module, dataloaders=data_module)


def train_n_trials(args):
    """This function trains a model parametrized by the configuration in args
    n times (also specified in args).
    As a result, writes a .csv in args.trials_results with the following
    columns for each trial:

        * Train accuracy and loss
        * Validation accuracy and loss
        * Number of epochs
        * Path of checkpoint
    """
    trial_args = copy.deepcopy(args)
    metrics_list = list()
    for trial in range(args.n_trials):
        trial_args.run_id = args.run_id + "_trial_" + str(trial)
        trial_checkpoint_file = os.path.join(trial_args.ckpt_folder, trial_args.run_id) + ".ckpt"

        my_callbacks = prepare_callbacks(trial_args)
        trainer = pl.Trainer.from_argparse_args(trial_args.as_namespace(), callbacks=list(my_callbacks.values()))

        module = VideoClassifier(**trial_args)  # we pass the unpacked (**args) dictionary to be able to save each hyperparam
        data_module = DataModule(trial_args)

        trainer.fit(module, datamodule=data_module)

        # extract epoch and train acc from best_model_path because we don't
        # know how to do it properly
        best_model_path = my_callbacks["checkpoint_callback"].best_model_path

        epochs = int(best_model_path.split("epoch=")[1].split("-")[0])
        train_acc = float(best_model_path.split("train_acc=")[1].split("-")[0])

        # validate the best model
        trial_metrics = trainer.validate(dataloaders=data_module)

        # save the metrics
        to_append = copy.deepcopy(trial_metrics[0])  # we only have one dataloader
        to_append["epochs"] = epochs
        to_append["train_acc"] = train_acc
        metrics_list.append(to_append)

        trainer.save_checkpoint(trial_checkpoint_file)

    pd.DataFrame(metrics_list).to_csv(args.trials_results_dest)
