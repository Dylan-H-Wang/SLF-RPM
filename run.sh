#!/bin/bash

# MAHNOB-HCI
python main.py --gpu 0 --epochs 150 --batch_size 128 --lr 1e-4 \
		--n_dim 2048 --temperature 1 \
		--dataset_name "mahnob-hci" \
		--dataset_dir "path/to/dataset" \
		--workers 4 --vid_frame 150 --clip_frame 30 --roi_list 0 1 2 3 4 5 6 --stride_list 1 2 3 4 5 \
		--log_dir "./logs/mahnob/train" \
		--model_depth 18

python test.py --gpu 1 --epochs 100 --batch_size 128 --lr 5e-3 \
	--pretrained "./logs/mahnob/train/best_train_model.pth.tar" \
	--dataset_name "mahnob-hci" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/mahnob/test" \
	--model_depth 18

# VIPL-HR-V2
python main.py --gpu 0 --epochs 200 --batch_size 128 --lr 1e-5 \
	--n_dim 512 --temperature 1 \
	--dataset_name "vipl-hr-v2" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --clip_frame 30 --roi_list 0 1 2 3 4 5 6 --stride_list 1 2 3 4 5 \
	--log_dir "./logs/vipl/train" \
	--model_depth 18

python test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 \
	--pretrained "./logs/vipl/train/best_train_model.pth.tar" \
	--dataset_name "vipl-hr-v2" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/vipl/test" \
	--model_depth 18

# UBFC
python main.py --gpu 0 --epochs 50 --batch_size 128 --lr 1e-4 \
	--n_dim 2048 --temperature 0.1 \
	--dataset_name "ubfc-rppg" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --clip_frame 30 --roi_list 0 1 2 3 4 5 6 --stride_list 1 2 3 4 5 \
	--log_dir "./logs/ubfc/train" \
	--model_depth 18

python test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 --dropout 0 \
	--pretrained "./logs/ubfc/train/best_train_model.pth.tar" \
	--dataset_name "ubfc-rppg" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/ubfc/test" \
	--model_depth 18