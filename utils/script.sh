# download dataset
pip3 install gdown
gdown --id 1v1z7NRyF9wD6wAlZBbphBZgTuIs8zOas
unzip -o -d ./ Allweather_subset.zip
# train net
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train_gpu_swtrans.py --useddp 1
# visualization model and metrics
tensorboard --logdir=/home/chenhui/PycharmProjects/Deeplearning/Transweather/logs/tensorboard/2022-10-29_15:14:33/
# resume model
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train_gpu_swtrans.py --useddp 1 --num_epochs 1 --pretrained 1 --isresume 1 time_str 2022-10-29_15:14:33

python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train_gpu_swtrans.py --useddp 1

python3 train_gpu_swtrans.py --train_batch_size=1 --val_batch_size=1 --num_epochs=50

python3 train_gpu_swtrans.py --train_batch_size=32 --val_batch_size=32 --num_epochs=2 --pretrained 1 --isresume 1 --time_str 2022-10-31_15:21:31