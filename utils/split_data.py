import glob
import os
import shutil
from PIL import Image
from tqdm.notebook import tqdm_notebook

def split_data(num_files_per_class, 
               origin_path,
               dest_path,
               resolution=512):

    NUM_FILES_PER_CLASS = num_files_per_class
    ORIGIN_PATH = origin_path

    # ORIGIN_PATH = f"/home/flix/Documents/oct-data/CellData/OCT/train"
    DEST_PATH = f"/{dest_path}/data_{NUM_FILES_PER_CLASS}_{resolution}"


    class_directories = glob.glob(f"{ORIGIN_PATH}/*")

    for class_dir in class_directories:
        curr_class = class_dir.split("/")[-1]
        if not os.path.isdir(f"{DEST_PATH}/{curr_class}"):
            os.makedirs(f"{DEST_PATH}/{curr_class}")
    if not os.path.isdir(f"{DEST_PATH}/UPLOAD"):
        os.makedirs(f"{DEST_PATH}/UPLOAD")

    for class_dir in class_directories:
        curr_class = class_dir.split("/")[-1]
        print(f"Current class: {curr_class}")
        files = glob.glob(class_dir + "/*")
        print(f"Number of available files: {len(files)}")
        
        for i, orig_file in tqdm_notebook(enumerate(files[:NUM_FILES_PER_CLASS]), total=NUM_FILES_PER_CLASS):

            filename = orig_file.split("/")[-1]
            img = Image.open(orig_file)
            img = img.resize((resolution,resolution))
            dest_file = DEST_PATH + f"/{curr_class}/{curr_class} ({i}).png"
            img.save(dest_file)

            img = Image.open(orig_file)
            img = img.resize((resolution,resolution))
            dest_file = DEST_PATH + f"/UPLOAD/{curr_class} ({i}).png"
            img.save(dest_file)
        