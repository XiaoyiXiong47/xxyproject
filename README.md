### Step 1: Prepare environment
Install all packages: run the following at /xxyproject/
`pip install -r requirements.txt`
Also install requried packages for Siformer at /xxyproject/slr-model/Siformer/



### Step 2: Predict gloss

Convert mp4 to csv format: (change *video_directory* or *output_filename* if needed)
`python D:\project_codes\xxyproject\utils\data_preprocess.py`

Then, run the following under ./slr-model/Siformer/
`python D:\project_codes\xxyproject\slr-model\Siformer\predict.py --model_path D:\project_codes\xxyproject\slr-model\Siformer\out-checkpoints/WLASL100v3/checkpoint_t_10.pth --csv_path D:\project_codes\xxyproject\slr-model\Siformer\datasets\processed_testing.csv`
The predicted label and corresponding video name will be saved at *D:\project_codes\xxyproject\data\predicted_label\predicted_gloss.csv*


### Step 3: Automatically annotate
`cd ../../models` go to /xxyproject/model
Run annotation script:
`python ./run_all_videos.py`
which automatically runs the following command for all videos:
`python ./models/auto_annotation.py --data_path /PATH/TO/DATASET `




### Siformer
Siformer was trained by : `python -m train --experiment_name WLASL100v3 --training_set_path datasets/wlasl100_train_v3.csv --validation_set_path datasets/wlasl100_val_v3.csv --validation_set from-file --num_classes 100 
--num_worker 4`
Dataset contains 1013 videos in total. Split 80-20 into training and testing datasets (**wlasl100_train_v3.csv** and **wlasl100_val_v3.csv**).

### Annotated xml files
The generated **xml files** in SLAML format are stored under **./data/annotations**.


### Result files 
Results for evaluation are under ./scripts/
(**result-keyframes.json**, **result-location.json**, **result_joint_angles.json**, **result_orientation.json**)

Excute **evaluate.py** for evaluation result.