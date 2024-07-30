from pathlib import Path
import os
import csv
import json
from typing import List


class CSV_Log_agent:
    """
    Class for creating agent for csv logging.
    """

    def __init__(self, l_col_names: List[str], FileDirectory: str, FileName: str) -> None:
        """
        Class for logging any set of individs` parameters.

        :param l_col_names: List of logging parameters names
        :param FileDirectory: Directory of log-file
        :param FileName: Name of log-file
        """
        self.full_filename = Path(FileDirectory) / Path(FileName)
        self.index = l_col_names

        # Preliminary process
        os.makedirs(FileDirectory, exist_ok=True)  # Check and create path with log-file
        self.file = open(self.full_filename, 'w+')  # Open file for writing
        self.file_writer = csv.writer(self.file, delimiter=";", lineterminator="\r")  # Determine line-writer
        self.file_writer.writerow(l_col_names)  # Write heading names as 1-st string

    def add_sol(self, l_param: list) -> None:
        """
        Function for logging one string of data.

        :param l_param: List of parameters to log
        """
        self.file_writer.writerow(l_param)

    def __del__(self) -> None:
        self.file.close()  # Close file


def JSON_Create(diction: dict, FileDirectory: str, FileName: str) -> None:
    """
    Function for creating JSON log-file with dictionary.

    :param diction: Dictionary for writing
    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file. Should be ended with ".txt"
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    os.makedirs(FileDirectory, exist_ok=True)  # Creating / checking existing of file-path
    with open(filename, 'w') as f:
        json.dump(diction, f, indent=4)  # Writing file


def JSON_Read(FileDirectory: str, FileName: str) -> dict:
    """
    Function for loading dictionary from log-file.

    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    with open(filename) as f:
        return json.load(f)  # Loading dictionary
