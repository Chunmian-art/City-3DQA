time=$(date "+%m%d%H%M")
tag=urban_mode
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --use_color --tag ${tag} --no_reference --no_lang_cls  > /workspace/UrbanQA/log/${time}_only_graph_${tag}.txt  2>&1 &