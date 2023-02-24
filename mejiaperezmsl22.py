"""mejiaperezmsl dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
from skeleton_graph import tssi_mejiaperez


BODY_JOINTS = ["pose_12", "pose_11", "pose_14", "pose_13", "root"]
FACE_JOINTS = ["face_249", "face_374", "face_382", "face_386", "face_7",
               "face_145", "face_155", "face_159", "face_13", "face_14",
               "face_324", "face_78", "face_276", "face_283", "face_282",
               "face_295", "face_46", "face_53", "face_52", "face_65"]
HAND_LEFT_JOINTS = ["leftHand_" + str(i) for i in range(21)]
HAND_RIGHT_JOINTS = ["rightHand_" + str(i) for i in range(21)]
BODY_PARTS = [BODY_JOINTS, FACE_JOINTS,
              HAND_RIGHT_JOINTS, HAND_LEFT_JOINTS]


# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
MejiaPerezMSL22 is a 22-label video dataset for Isolated Mexican Sign Language (ASL) recognition.
It contains 2,200 poses each one performed 25 times by four different people at different speeds, starting and ending times.
Each sign is composed of 20 consecutive frames containing hands, body, and facial 3D keypoint coordinates.
"""

# BibTeX citation
_CITATION = """
@article{mejia2022automatic,
  title={Automatic recognition of Mexican Sign Language using a depth camera and recurrent neural networks},
  author={Mej{\'\i}a-Per{\'e}z, Kenneth and C{\'o}rdova-Esparza, Diana-Margarita and Terven, Juan and Herrera-Navarro, Ana-Marcela and Garc{\'\i}a-Ram{\'\i}rez, Teresa and Ram{\'\i}rez-Pedraza, Alfonso},
  journal={Applied Sciences},
  volume={12},
  number={11},
  pages={5523},
  year={2022},
  publisher={MDPI}
}
"""


class MejiaPerezMsl22(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mejiaperezmsl22 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    INFO = pd.read_csv(
        "./train_validation_info.csv",
        index_col=0
    )
    DATA_URL = "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/MejiaPerezMSL.zip"
    TSSI_ORDER = tssi_mejiaperez()[1]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'pose': tfds.features.Tensor(shape=(20, len(self.TSSI_ORDER), 3), dtype=np.float64),
                'label': tfds.features.ClassLabel(names=list(self.INFO["label"].unique()))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('pose', 'label'),  # Set to `None` to disable
            homepage='https://github.com/ICKMejia/Mexican-Sign-Language-Recognition',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        # originally: path = dl_manager.download_and_extract('https://todo-data-url')
        path = dl_manager.download_and_extract(self.DATA_URL)

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train_1": self._generate_examples(path, 'train', cv_split=1),
            "train_2": self._generate_examples(path, 'train', cv_split=2),
            "train_3": self._generate_examples(path, 'train', cv_split=3),
            "train_4": self._generate_examples(path, 'train', cv_split=4),
            "train_5": self._generate_examples(path, 'train', cv_split=5),
            "validation_1": self._generate_examples(path, 'validation', cv_split=1),
            "validation_2": self._generate_examples(path, 'validation', cv_split=2),
            "validation_3": self._generate_examples(path, 'validation', cv_split=3),
            "validation_4": self._generate_examples(path, 'validation', cv_split=4),
            "validation_5": self._generate_examples(path, 'validation', cv_split=5),
            "train": self._generate_examples(path, 'train'),
            "validation": self._generate_examples(path, 'validation'),
            "test": self._generate_examples(path, 'test')
        }

    def _generate_examples(self, source_path, split, cv_split=None):
        """Generator of examples for each split."""
        if split == "test":
            path = source_path / "Testing"
        elif split == "train" or split == "validation":
            path = source_path / "TrainingValidation"
        else:
            raise Exception(f"Unknown split {split}")

        if split == "test":
            for filepath in path.glob('*.csv'):
                # Yields (key, example)
                split_by_underscore = filepath.name.split("_")
                label = "_".join(split_by_underscore[:-2])
                yield filepath.name, {
                    'pose': self.convert_csv_to_tssi(filepath,
                                                     self.TSSI_ORDER,
                                                     data_bounds=[0, 2.5],
                                                     scale_to=[0, 1]),
                    'label': label
                }
        elif cv_split is None:
            filenames = self.get_split_filenames(split)
            for filename in filenames:
                # Yields (key, example)
                split_by_underscore = filename.split("_")
                label = "_".join(split_by_underscore[:-2])
                yield filename, {
                    'pose': self.convert_csv_to_tssi(path / filename,
                                                     self.TSSI_ORDER,
                                                     data_bounds=[0, 2.5],
                                                     scale_to=[0, 1]),
                    'label': label
                }
        else:
            filenames = self.get_cv_split_filenames(split, cv_split-1)
            for filename in filenames:
                # Yields (key, example)
                split_by_underscore = filename.split("_")
                label = "_".join(split_by_underscore[:-2])
                yield filename, {
                    'pose': self.convert_csv_to_tssi(path / filename,
                                                     self.TSSI_ORDER,
                                                     data_bounds=[0, 2.5],
                                                     scale_to=[0, 1]),
                    'label': label
                }

    def get_split_filenames(self, split):
        sss = StratifiedShuffleSplit(n_splits=1,
                                    test_size=0.1,
                                    random_state=0)
        filenames = self.INFO["filename"]
        labels = self.INFO["label"]
        splits = list(sss.split(filenames, labels))
        train_indices, test_indices = splits[0]
        if split == "train":
            filenames = self.INFO.loc[train_indices, "filename"]
            return filenames
        else:
            filenames = self.INFO.loc[test_indices, "filename"]
            return filenames
    
    def get_cv_split_filenames(self, split, cv_split):
        skf = StratifiedKFold(n_splits=5,
                              random_state=0,
                              shuffle=True)
        filenames = self.INFO["filename"]
        labels = self.INFO["label"]
        splits = list(skf.split(filenames, labels))
        train_indices, test_indices = splits[cv_split-1]
        if split == "train":
            filenames = self.INFO.loc[train_indices, "filename"]
            return filenames
        else:
            filenames = self.INFO.loc[test_indices, "filename"]
            return filenames

    
    def convert_csv_to_tssi(self, filepath, columns_in_tssi_order=None,
                            data_bounds=None, scale_to=None):
        # Read the file
        df = pd.read_csv(filepath, index_col=0)

        # Rename columns
        i = 0
        mapper = {}
        for part in BODY_PARTS:
            for axis in ["x", "y", "z"]:
                for joint in part:
                    old_col = df.columns[i]
                    new_col = joint + "_" + axis
                    mapper[old_col] = new_col
                    i = i + 1

        df = df.rename(columns=mapper)

        # Sort columns in tssi order
        if columns_in_tssi_order is None:
            columns_in_tssi_order = BODY_JOINTS + FACE_JOINTS \
                + HAND_RIGHT_JOINTS + HAND_LEFT_JOINTS
        columns_x = [f"{col}_x" for col in columns_in_tssi_order]
        columns_y = [f"{col}_y" for col in columns_in_tssi_order]
        columns_z = [f"{col}_z" for col in columns_in_tssi_order]
        
        # Create numpy matrix
        matrix = np.zeros((len(df), len(columns_x), 3))
        matrix[:, :, 0] = df.loc[:, columns_x].to_numpy()
        matrix[:, :, 1] = df.loc[:, columns_y].to_numpy()
        matrix[:, :, 2] = df.loc[:, columns_z].to_numpy()

        # Shift to positive bound
        x_min = matrix[:, :, 0].min()
        y_min = matrix[:, :, 1].min()
        if x_min < 0:
            matrix[:, :, 0] = matrix[:, :, 0] + np.abs(x_min)
        if y_min < 0:
            matrix[:, :, 1] = matrix[:, :, 1] + np.abs(y_min)

        # Scale to (0, 1)
        if data_bounds and scale_to:
            data_min, data_max = data_bounds or [0, 1]
            range_min, range_max = scale_to or [0, 1]
            std = (matrix - data_min) / (data_max - data_min)
            matrix = std * (range_max - range_min) + range_min

        return matrix

