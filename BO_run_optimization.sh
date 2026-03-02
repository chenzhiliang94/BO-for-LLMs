#!/bin/bash

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=1

# ----------------------- #
# Shared configuration
# ----------------------- #
ITER=50
NUM_DATA=10000
EPOCHS=1
TRIALS=1
EXP_SETTING=in_dist
TIME_LIMIT=1000
LORA_RANK=128
NUM_EVAL_SAMPLES=5
TRAIN_BATCH=4
EVAL_BATCH=4
OUTPUT_DIR=results
PRINTOUT_DIR=printouts
USE_JOBS=0
UCB_BETA=20

# ----------------------- #
# Sweep variables
# ----------------------- #
OPT_METHODS=("mixed")
ACQ_FUNCS=("ucb")
EVAL_METHODS=("eval_loss")
RUN_BO_ON_OPTIONS=("both")
MODELS=("llama-8b")
TRAINING_TASKS_OPTIONS=("triviaqa,truthfulqa_gen,gsm8k")

# evaluation tasks
TASKS=("gsm8k")

# Track failures
FAILED_JOBS=()

# ----------------------- #
# Create output directories
# ----------------------- #
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PRINTOUT_DIR"

# ----------------------- #
# Function to run a job
# ----------------------- #

run_job() {
    local task=$1
    local opt_method=$2
    local acq_func=$3
    local eval_method=$4
    local run_bo_on=$5
    local model=$6
    local training_tasks=$7
    local seed=13549

    # create a random run id with random
    run_id=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 8)

    INFO_PRINTOUT="${opt_method}_acq_${acq_func}_eval_${eval_method}_bo_${run_bo_on}_run_id_${run_id}"
    SAVE_NAME="${model}_${acq_func}_${INFO_PRINTOUT}.json"
    LOG_FILE="${PRINTOUT_DIR}/${model}_${acq_func}_${task}_${INFO_PRINTOUT}.out"

    echo "==============================================="
    echo "MODEL=$model"
    echo "RUN_BO_ON=$run_bo_on"
    echo "TASK=$task"
    echo "TRAINING_TASKS=$training_tasks"
    echo "OPT_METHOD=$opt_method"
    echo "ACQ_FUNC=$acq_func"
    echo "EVAL_METHOD=$eval_method"
    echo "OUTPUT AT ${LOG_FILE}"
    # the task, make it comma separated into separated by _
    # and add it to the print here
    echo "RESULTS WILL BE SAVED AT ${OUTPUT_DIR}/${task//,/_}/${SAVE_NAME}"
    echo "==============================================="

    python3 -u BO_runs_LLM_joint_optimization.py \
        --iterations=$ITER \
        --num_data=$NUM_DATA \
        --epochs=$EPOCHS \
        --trials=$TRIALS \
        --eval_tasks=$task \
        --training_tasks=$training_tasks \
        --experiments_setting=$EXP_SETTING \
        --time_limit=$TIME_LIMIT \
        --lora_rank=$LORA_RANK \
        --num_eval_samples=$NUM_EVAL_SAMPLES \
        --run_BO_on=$run_bo_on \
        --training_batch=$TRAIN_BATCH \
        --evaluation_batch=$EVAL_BATCH \
        --eval_method=$eval_method \
        --seed=$seed \
        --acq_function=$acq_func \
        --model=$model \
        --JoBS=$USE_JOBS \
        --ucb_beta=$UCB_BETA \
        --optimize_method=$opt_method \
        --output_dir=$OUTPUT_DIR \
        --save_name="$SAVE_NAME" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR in job: $model | $run_bo_on | $task | $opt_method | $acq_func | $eval_method"
        echo "   Check log: $LOG_FILE"
        FAILED_JOBS+=("$model | $run_bo_on | $task | $opt_method | $acq_func | $eval_method")
    else
        echo "✅ DONE: $model | $run_bo_on | $task | $opt_method | $acq_func | $eval_method"
    fi

    echo ""
}

# ----------------------- #
# Sweep loop
# ----------------------- #

for model in "${MODELS[@]}"; do
    for run_bo_on in "${RUN_BO_ON_OPTIONS[@]}"; do
        for training_tasks in "${TRAINING_TASKS_OPTIONS[@]}"; do
            for task in "${TASKS[@]}"; do
                for opt_method in "${OPT_METHODS[@]}"; do
                    for acq_func in "${ACQ_FUNCS[@]}"; do
                        for eval_method in "${EVAL_METHODS[@]}"; do
                            run_job "$task" "$opt_method" "$acq_func" "$eval_method" "$run_bo_on" "$model" "$training_tasks"
                        done
                    done
                done
            done
        done
    done
done

# ----------------------- #
# Final summary
# ----------------------- #

echo "==============================================="
echo "Unit Test Summary"
echo "==============================================="

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "🎉 All jobs completed successfully!"
else
    echo "❌ ${#FAILED_JOBS[@]} job(s) failed:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "   - $job"
    done
fi
