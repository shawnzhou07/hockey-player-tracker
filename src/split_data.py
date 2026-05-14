import os # creates directories, handles file paths
import shutil # copies files
import random # shuffles lists

def split_dataset(annotations_dir, images_dir, train_dir, val_dir, split_ratio=0.8):
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')] # get all .txt annotation files
    
    random.shuffle(annotation_files) # shuffle randomly
    
    split_index = int(len(annotation_files) * split_ratio) # calculate split point
    
    train_files = annotation_files[:split_index] # first 80% for training
    val_files = annotation_files[split_index:] # remaining 20% for validation
    
    # copy train files
    for txt_file in train_files:
        img_file = txt_file.replace('.txt', '.jpg') # get corresponding image filename
        
        # copy annotation file
        shutil.copy(
            os.path.join(annotations_dir, txt_file),
            os.path.join(train_dir, 'labels', txt_file)
        )
        
        # copy image file
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(train_dir, 'images', img_file)
        )
    
    # copy val files
    for txt_file in val_files:
        img_file = txt_file.replace('.txt', '.jpg') # get corresponding image filename
        
        # copy annotation file
        shutil.copy(
            os.path.join(annotations_dir, txt_file), # source
            os.path.join(val_dir, 'labels', txt_file) # destination
        )
        
        # copy image file
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(val_dir, 'images', img_file)
        )
    
    print(f"Split complete: {len(train_files)} train, {len(val_files)} val")

# call the function
split_dataset(
    annotations_dir="data/annotations",
    images_dir="data/labeled_frames/TBL@BUF_2026-03-08",
    train_dir="data/train",
    val_dir="data/val"
)