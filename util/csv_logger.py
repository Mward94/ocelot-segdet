"""Class to log data to file. Data will be logged as a CSV"""
import csv
import os

from util.helpers import replace_extension


class CSVLogger:
    def __init__(self, filepath, overwrite=False):
        """Creates an empty file to log data to

        Args:
            filepath (str): Path to file to log data to. Extension will automatically be set to .csv
            overwrite (bool): Whether the CSV will be overwritten if it already exists
        """
        self.filepath = replace_extension(filepath, '.csv')

        # Handle file already existing
        if not overwrite and os.path.isfile(self.filepath):
            raise FileExistsError(
                f'Error. File \'{self.filepath}\' already exists. Set the overwrite flag to '
                f'overwrite the file')

        # Clear the file (ensuring it's empty)
        with open(self.filepath, 'w') as file:
            pass

        # Initialize the set of headers to None. These will be set the first time data is written
        self.headers = None
        self.headers_written = False

    def write_dict(self, dict_data):
        """Writes a dictionary to the CSV file

        Keys = column headers

        If this is the first time write_dict is called, headers are determined from the given data

        Args:
            dict_data (dict): Dictionary of data to be written to file
        """
        if self.headers is None:
            self.headers = list(dict_data.keys())

        # Adding newline='' gets rid of extra line on Windows
        with open(self.filepath, 'a', newline='') as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self.headers)
            if not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            writer.writerow(dict_data)
