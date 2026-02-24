#!/bin/bash

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

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
NUM_EVAL_SAMPLES=100
TRAIN_BATCH=8
EVAL_BATCH=8
OUTPUT_DIR=results
PRINTOUT_DIR=printouts
USE_JOBS=0
UCB_BETA=20

# ----------------------- #
# Sweep variables
# ----------------------- #

OPT_METHODS=("random" "multi_fidelity" "multi_fidelity_KG" "mixed")
ACQ_FUNCS=("ucb" "EI")
EVAL_METHODS=("eval_loss" "performance")
RUN_BO_ON_OPTIONS=("data")
MODELS=("llama-8b" "qwen-7b")

# evaluation tasks
TASKS=("triviaqa")

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
    local seed=13549

    INFO_PRINTOUT="_${opt_method}_acq_${acq_func}_eval_${eval_method}_bo_${run_bo_on}"
    SAVE_NAME="${model}_${acq_func}_${task}_${INFO_PRINTOUT}.json"
    LOG_FILE="${PRINTOUT_DIR}/${model}_${acq_func}_${task}_${INFO_PRINTOUT}.out"

    echo "==============================================="
    echo "MODEL=$model"
    echo "RUN_BO_ON=$run_bo_on"
    echo "TASK=$task"
    echo "OPT_METHOD=$opt_method"
    echo "ACQ_FUNC=$acq_func"
    echo "EVAL_METHOD=$eval_method"
    echo "OUTPUT AT ${LOG_FILE}"
    echo "RESULTS WILL BE SAVED AT ${OUTPUT_DIR}/${SAVE_NAME}"
    echo "==============================================="

    python3 -u BO_runs_LLM_joint_optimization.py \
        --iterations=$ITER \
        --num_data=$NUM_DATA \
        --epochs=$EPOCHS \
        --trials=$TRIALS \
        --eval_tasks=$task \
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
        for task in "${TASKS[@]}"; do
            for opt_method in "${OPT_METHODS[@]}"; do
                for acq_func in "${ACQ_FUNCS[@]}"; do
                    for eval_method in "${EVAL_METHODS[@]}"; do
                        run_job "$task" "$opt_method" "$acq_func" "$eval_method" "$run_bo_on" "$model"
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
echo "Summary of Data Optimization Runs"
echo "==============================================="

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "🎉 All jobs completed successfully!"
else
    echo "❌ ${#FAILED_JOBS[@]} job(s) failed:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "   - $job"
    done
fi
