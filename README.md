Install all packages:
`pip install -r requirements.txt`


Predict gloss:
run the following under ../SiFormer/
`python predict.py --model_path out-checkpoints/WLASL100/checkpoint_v_10.pth --csv_path D:\project_codes\xxyproject\uti
ls\testing_samples.csv`

Run annotation script:
`python ./models/run_all_videos.py`
which automatically runs the following command for all videos:
`python ./models/auto_annotation.py --data_path /PATH/TO/DATASET `



