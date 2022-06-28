import pandas as pd
from sklearn import metrics
import torch

from utils.dataset import DataModule


def compute_statistics(inp_df, args):
    """Computes the statistics of the results.
    Specifically, it shows the video statistics of the test and also a summary
    of the performance including the accuracy, precision, recall and fscore.
    Also, it shows the average inference times.
    """
    y_true = inp_df["target"]
    y_pred = inp_df["prediction"]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    class_recall = [metrics.recall_score(y_true, y_pred, labels=[i], average="micro") for i in range(args.num_classes)]
    class_precision = [metrics.precision_score(y_true, y_pred, labels=[i], average="micro") for i in range(args.num_classes)]
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)

    print("Confusion Matrix:")
    print("B, R, Sh, Sw")  # bike road shared sidewalk
    print(confusion_matrix)
    print("-" * 10)
    print("Class Recall", class_recall)
    print("Class Precision", class_precision)
    print("Overall Accuracy", overall_accuracy)


def load_scripted_model(args):
    """This function simply loads the scripted model (.pt) of the pipeline."""
    model = torch.jit.load(args.deployed_mobile)
    return model


def concat_and_extract(predictions, labels, video_names, clip_idxs):
    """This function takes a bunch of tensors with batch dimension and returns
    a list containing tuples for each element of the batch.
    The columns are: (video_name, clip_idx, predictions, labels)
    """
    zipped_lists = zip(video_names, clip_idxs.tolist(), predictions.tolist(), labels.tolist())
    return [[name, clip_idx, prediction, label] for name, clip_idx, prediction, label in zipped_lists]


def test_by_clips(model, data_loader, args):
    result = list()
    print("Batch number 0 of ", len(data_loader), "length of results is: ", len(result))
    for batch_idx, batch in enumerate(data_loader):
        print("Batch number", batch_idx, "length of results is: ", len(result))
        video = batch["video"]
        label = batch["label"]
        bs = video.size(0)
        prediction = torch.zeros(bs)
        for i in range(bs):
            model_output = model(video[i:i + 1])  # input has to have shape input blob size [1, 3, nframes, side, side]
            prediction[i] = torch.argmax(model_output).item()

        new_results = concat_and_extract(prediction, label, batch["video_name"], batch["clip_index"])

        result.extend(new_results)

    return pd.DataFrame(result, columns=["video_label", "clip_idx", "prediction", "target"])


def test_deployed(args):
    print("Testing deployed")
    data = DataModule(args)
    model = load_scripted_model(args)
    print("Loaded all we need")
    result_df = test_by_clips(model, data.test_dataloader(), args)
    result_df.to_csv(f"{args.results_folder}/{args.run_id}2.csv")
    compute_statistics(result_df, args)
