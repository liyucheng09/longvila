MODEL_NAME=$1
MODEL_PATH=$2
VIDEO_PATH=$3
prompt_template=qwen2
max_frame_num=3000
frame_interval=200

eval_path=llava/eval/vision_niah_vila
mkdir -p $eval_path/data/haystack_embeddings/$MODEL_NAME
mkdir -p $eval_path/data/needle_embeddings/$MODEL_NAME
python $eval_path/produce_haystack_embedding.py --model $MODEL_PATH --output_dir $eval_path/data/haystack_embeddings/$MODEL_NAME --sampled_frames_num $max_frame_num --pooling_size 0 --video_path $VIDEO_PATH
python $eval_path/produce_needle_embedding.py --model $MODEL_PATH --output_dir $eval_path/data/needle_embeddings/$MODEL_NAME --pooling_size 0 --needle_dataset LongVa/v_niah_needles

accelerate launch --num_processes 4 $eval_path/eval_vision_niah.py \
    --model  $MODEL_PATH \
    --needle_embedding_dir $eval_path/data/needle_embeddings/$MODEL_NAME \
    --haystack_dir $eval_path/data/haystack_embeddings/$MODEL_NAME \
    --needle_dataset lmms-lab/v_niah_needles \
    --prompt_template $prompt_template \
    --max_frame_num $max_frame_num \
    --min_frame_num  $frame_interval \
    --frame_interval $frame_interval \
    --depth_interval 0.2
