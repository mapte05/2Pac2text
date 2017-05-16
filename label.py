from __future__ import print_function
import csv, os

directory_path = "festival/utts"
for filename in os.listdir(directory_path):
    if filename.endswith(".utt"):
        with open(os.path.join(directory_path, filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')
            for i, row in enumerate(csv_reader):
                if i == 4:
                    sentence = row[2][10:-4]
                    with open(os.path.join("labels", filename), "w") as output_file:
                        output_file.write(sentence + '\n')
