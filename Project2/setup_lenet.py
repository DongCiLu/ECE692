git clone https://github.com/tensorflow/models.git

DATA_DIR=./cifar10
python models/slim/download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="${DATA_DIR}"
cp -r models/slim/datasets .
