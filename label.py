from __future__ import print_function
import csv, os
import sys


assert len(sys.argv)==2 # user supplies directory with utterances and transcriptions in the data/ dir
cmu_dir = sys.argv[1]

directory_path = "data/" + cmu_dir + "/festival/utts"
output_directory = "data/" + cmu_dir + "/labels"
file_extension = ".utt"

for filename in os.listdir(directory_path):
    if filename.endswith(file_extension):
        with open(os.path.join(directory_path, filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|')
            for i, row in enumerate(csv_reader):
                if i == 4:
                    sentence = row[2][10:-4]
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    with open(os.path.join(output_directory, filename), "w") as output_file:
                        output_file.write(sentence + '\n')
