import argparse
import pandas as pd
import os
from itertools import repeat
from sklearn.model_selection import train_test_split as split_df

CLASSDICT = {
    "bike_bidir": 0,
    "bike_unidir": 0,
    "road": 1,
    "shared": 2,
    "sidewalk": 3
}

CLASSDICT21 = {
    "sidewalk": 3,
    "road": 1,
    "shared": 2,
    "BikeBi": 0,
    "BikeU": 0
}


def tri_split(input_df: pd.DataFrame, partition_ratios: tuple, stratify_by=None, random_seed=1234):
    """This function partitions the input_df into three different dataframes
    with the ratios specified in partition_ratios. Also it is stratified by the colums.
    It means that data will be distributed according with this value.

    Returns: pd.DataFrame, pd.DataFrame, pd.DataFrame
    """
    # first split the train test
    stratify_column1 = input_df[stratify_by] if stratify_by is not None else None  # select the column by which we will stratify.
    train_df, rest_df = split_df(input_df, train_size=partition_ratios[0], stratify=stratify_column1, random_state=random_seed)

    if partition_ratios[1] == 0:
        empty_df = pd.DataFrame({"path": [], "target": []})
        return train_df, empty_df, rest_df

    # the proportion of the rest that goes into validation
    rebuild_partition = partition_ratios[1] * (1 / (partition_ratios[1] + partition_ratios[2]))

    stratify_column2 = rest_df[stratify_by] if stratify_by is not None else None  # select the column by which we will stratify.
    val_df, test_df = split_df(rest_df, train_size=rebuild_partition, stratify=stratify_column2, random_state=random_seed)

    return train_df, val_df, test_df


def get_all_mp4(directory: str) -> list:
    """This function returns a list of all the .mp4 files in a directory. Also
    it returns the full absolute path linking the filename with the directory.
    """
    if os.path.isdir(directory):
        simple_filenames = filter(lambda x: x.endswith(".mp4"), os.listdir(directory))  # first filter all .mp4 files
        return list(map(lambda x: os.path.join(directory, x), simple_filenames))  # then we add the prefix
    else:
        return list()  # return an empty list as it does not exist


def extract_uniques(inp_df):
    """This function filters the inp_df selecting the clips that whose name
    ends with _0.mp4. This is a simple way to select the unique identifiers
    (removing the clip index).
    """
    new_df = inp_df.copy()
    return new_df[new_df["path"].apply(lambda x: x.endswith("_0.mp4"))]


def extend_uniques(df_to_extend, full_df):
    """This function filters the full_df by selecting the clips contained in
    df_to_extend (that contains only unique names)
    """
    extended = full_df.copy()
    videos_in_partition = set(df_to_extend["path"].apply(lambda x: x.strip("_0.mp4")))
    return extended[extended["path"].apply(lambda x: "_".join(x.split("_")[:-1])).isin(videos_in_partition)]


def read_video_names_from_single_dict(root_directory: str) -> pd.DataFrame:
    """Reads all the .mp4 files of a root directory with structure as in
    CLASSDICT or CLASSDICT21. Can contain extra folders/files that will be ignored.

    Returns a dataframe with the video file path and the label
    """
    base_df = {"path": [], "target": []}
    all_classes = CLASSDICT | CLASSDICT21
    for label, label_id in all_classes.items():
        class_directory = os.path.join(root_directory, label)
        all_files = get_all_mp4(class_directory)
        all_labels = repeat(label_id, len(all_files))

        base_df["path"].extend(all_files)
        base_df["target"].extend(all_labels)

    return pd.DataFrame(base_df)


def read_all_video_names(root_directories: list) -> pd.DataFrame:
    """This function reads all the .mp4 files inside each directory in root_directories."""
    list_of_df = list()
    for root_dir in root_directories:
        dir_df = read_video_names_from_single_dict(root_dir)
        list_of_df.append(dir_df)

    return pd.concat(list_of_df, axis=0)


def main(args):
    full_df = read_all_video_names(args.root_directories)
    full_df.to_csv(f"{args.folder_destination}/full_data_2022.csv")

    uniques_dataset = extract_uniques(full_df)
    train_df_uniq, validation_df_uniq, test_df_uniq = tri_split(uniques_dataset, (0.6, 0.2, 0.2), stratify_by="target", random_seed=args.random_seed)
    print("Partition done")

    train_df = extend_uniques(train_df_uniq, full_df)
    print("Train len", len(train_df))
    train_df.to_csv(f"{args.folder_destination}/train.csv", index=False, sep=" ", header=False)

    validation_df = extend_uniques(validation_df_uniq, full_df)
    print("Validation len", len(validation_df))
    validation_df.to_csv(f"{args.folder_destination}/validation.csv", index=False, sep=" ", header=False)

    test_df = extend_uniques(test_df_uniq, full_df)
    print("Test len", len(test_df))
    test_df.to_csv(f"{args.folder_destination}/test.csv", index=False, sep=" ", header=False)

    print("Partition saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-root", "--root-directories", type=str, nargs="+", help="The path of the directory where the videos are stored. Different folders with the classes are expected.")

    parser.add_argument("-dest", "--folder-destination", type=str, help="The folder in which the full dataset will be saved.")

    parser.add_argument("-partition", "--partition-data", type=bool, action=argparse.BooleanOptionalAction, chelp="If this flag is set to true, the dataset will be partitioned into train, validation and test subsets with 60, 20, 20 proportion.")

    parser.add_argument("-seed", "--random-seed", type=int, default=1234)

    args = parser.parse_args()

    main(args)
