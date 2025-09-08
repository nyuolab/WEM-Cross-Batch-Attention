export WANDB_API_KEY=[REPLACE WITH API KEY]
export WANDB_ENTITY=[REPLACE WITH ACCOUNT]
export OMP_NUM_THREADS=32

export tokenizer_suffix="4k"
export position_encoding='rope'


# Maximum number of retries in case of failure.
retries=0
max_retries=1

# Load enroot module if required
module load enroot
export XDG_RUNTIME_DIR=#LOCAL_DIR#
export XDG_CONFIG_HOME=#LOCAL_DIR#/.config
export XDG_CACHE_HOME=#LOCAL_DIR#/.cache
export XDG_DATA_HOME=#LOCAL_DIR#/.local/share
GIVEN_NAME=# NAME OF ENROOT CONTAINER
GIVEN_SOURCE=#LOCAL_DIR#/cuda_12.8_torch_2.5.1.sqsh

if ! enroot list | grep -q "${GIVEN_NAME}"; then
    echo "Enroot image '${GIVEN_NAME}' not found. Importing from ${GIVEN_SOURCE}"
    enroot create --name "${GIVEN_NAME}" "${GIVEN_SOURCE}"
else
    echo "Enroot image '${GIVEN_NAME}' found."
fi

while [ $retries -lt $max_retries ]; do
    # Not actually sure why you need the srun here? 
    enroot start -r -w \
      --mount #LOCAL_DIR#/WEM-Cross-Batch-Attention/:/workspace/Whole_Exome_Modeling/ \
      --mount #CHECKPOINT DIR#:/workspace/checkpoints/ \
      -e WANDB_API_KEY=$WANDB_API_KEY \
      -e OMP_NUM_THREADS=$OMP_NUM_THREADS \
      -e WANDB_ENTITY=$WANDB_ENTITY \
      -e WORLD_SIZE=$WORLD_SIZE \
      -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
      NAME_OF_ENROOT_CONTAINER \
      -- bash -c "\
        export PYTHONPATH="/workspace/Whole_Exome_Modeling:$PYTHONPATH" && \
        torchrun \
          --standalone \
          --nproc-per-node=8 \
          /workspace/Whole_Exome_Modeling/train/train_encoder_WEM.py \
          --n_head 16 \
          --n_embd 1024 \
          --n_layer 12 \
          --num_genes 256 \
          --sequence_length 2048 \
          --save_freq 500000000 \
          --mini_batch_size 1 \
          --token_budget 100e9 \
          --batch_size 32 \
          --warmup_period 0.004 \
          --lr 0.05 \
          --checkpoint_freq 2 \
          --save_name /workspace/checkpoints/wem_checkpoints/WEM-DOLMA-100M-${position_encoding}-${tokenizer_suffix} \
          --wandb_project_name \"WEM-TESTING\" \
          --base_dir /workspace/data \
          --dropout 0.05 \
          --position_encoding ${position_encoding}  \
          --tokenizer /workspace/tokenizers/dolma_4k.model \
          --beta1 0.9 \
          --beta2 0.95 \
          --test_freq 100000000 \
          --dataloader \"random\" \
          --tokenizer_suffix ${tokenizer_suffix} \
          --compile
      "
    
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
      echo "Job completed successfully."
      break
    else
      ((retries++))
      echo "Job crashed with exit status $exit_status. Attempt $retries of $max_retries."
      sleep 15  # Pause before retrying.
    fi
done

if [ $retries -ge $max_retries ]; then
  echo "Max retries reached. Exiting."
fi
