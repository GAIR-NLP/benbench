
export HF_ENDPOINT=https://hf-mirror.com

chatglm_path=/data1/ckpts/chatglm-6b/
chatglm2_path=/data/ckpts/chatglm2/6b/models--THUDM--chatglm2-6b/snapshots/8fd7fba285f7171d3ae7ea3b35c53b6340501ed1
chatglm3_path=/data/ckpts/chatglm3/6b-chat/models--THUDM--chatglm3-6b/snapshots/fc3235f807ef5527af598c05f04f2ffd17f48bab

# python cal_n_gram_acc.py --model_path ${chatglm_path} \
#     --gpu_id 0 --n 3 --k 5 \
#     --multi_processes --num_processes 24

python cal_n_gram_acc.py --model_path ${chatglm_path} \
    --gpu_id 0 --n 3 --k 5 
