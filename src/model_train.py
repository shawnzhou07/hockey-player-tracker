from ultralytics import YOLO

model = YOLO('yolov8n.pt') # load pre-trained yolo model

# train on our hockey data
results = model.train(
    data='data/dataset.yaml',
    epochs=50, # how many times the model sees every training image
    imgsz=640, # resizes to square, 640 x 640 in this case
    batch=16, # how many images processed at a time
    patience=10, # stops early if there is no improvement after a number of epochs
    name='hockey_player_arena_detector'
)

print("Training complete!")