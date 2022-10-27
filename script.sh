# download dataset
pip3 install gdown
gdown --id 1v1z7NRyF9wD6wAlZBbphBZgTuIs8zOas
unzip -o -d ./ Allweather_subset.zip
# train net
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 swmain.py --useddp 1
