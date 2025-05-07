import os
import shutil
import random

with open("data/all_list.txt", "w") as f:
    img_dir = "data/imgs"
    for filename in os.listdir(img_dir):
        f.write(filename + "\n")


with open("data/all_list.txt", "r") as f:
    lines = f.readlines()

for i in range(5):
    random.shuffle(lines)
    with open("data/train_new" + str(i) + ".txt", "w") as f:
        for line in lines[:int(len(lines)*0.8)]:
            f.write(line)

    with open("data/valid_new" + str(i) + ".txt", "w") as f:
        for line in lines[int(len(lines)*0.8):]:
            f.write(line)
