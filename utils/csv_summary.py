import csv
import os

class CSVSummary():
    def __init__(self, folder,file_format = ["x", "y"]):
        self.folder = folder
        self.file_format = file_format
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


    def write(self, param, value):
        # value should be a list of all variables u want to write in a one row
        full_path = self.folder + "/" + param + ".csv"
        if not os.path.exists(full_path):
            f = open(full_path, 'w+')
            writer = csv.writer(f, lineterminator='\n', delimiter=',')
            writer.writerow(self.file_format)
            print(full_path + " created")
            f.close()

        with open(full_path, "a") as f:
            writer = csv.writer(f, lineterminator='\n', delimiter=',')
            writer.writerow(value)

