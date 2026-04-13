#!/bin/bash

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# ----------------------- #
# Shared configuration
# ----------------------- #
ITER=50
NUM_DATA=10000
EPOCHS=1
TRIALS=2
EXP_SETTING=in_dist
TIME_LIMIT=1000
LORA_RANK=128
NUM_EVAL_SAMPLES=200
TRAIN_BATCH=36
EVAL_BATCH=36
RESULTS_ROOT=results
PRINTOUT_DIR=printouts
USE_JOBS=0
# Representative JoBS predictors to sweep.
JOBS_PREDICTOR_MODELS=()
UCB_BETA=20
COST_SCALE_MF=1
NUM_INITIAL_RANDOM_SAMPLES=10

# ----------------------- #
# Sweep variables
# ----------------------- #
OPT_METHODS=("mixed")
ACQ_FUNCS=("pes")
EVAL_METHODS=("eval_loss")
RUN_BO_ON_OPTIONS=("both")
MODELS=("llama-8b")
TRAINING_TASKS_OPTIONS=("truthfulqa_gen,commonsense_qa,mmlu,triviaqa,gsm8k,arc_challenge")

# evaluation tasks
TASKS=( "triviaqa" "arc_challenge" "commonsense_qa" "mmlu" "truthfulqa_gen" "gsm8k")


# Track failures
FAILED_JOBS=()

# ----------------------- #
# Create output directories
# ----------------------- #
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
    local jobs_predictor_model=$8
    local seed=12345

    # Output dir based on what BO optimizes: results_both, results_model, results_data
    local output_dir="${RESULTS_ROOT}_${run_bo_on}"
    mkdir -p "$output_dir"

    # Log dir with run_bo_on subdirectory
    local log_dir="${PRINTOUT_DIR}/${run_bo_on}"
    mkdir -p "$log_dir"

    local jobs_suffix=""
    if [ "$USE_JOBS" -eq 1 ]; then
        jobs_suffix="_jobs_${jobs_predictor_model}"
    fi

    SAVE_NAME="${model}_${acq_func}_${opt_method}_eval_${eval_method}${jobs_suffix}_seed_${seed}.json"
    LOG_FILE="${log_dir}/${model}_${acq_func}_${task}_${opt_method}_eval_${eval_method}${jobs_suffix}_seed_${seed}.out"

    # Skip if results file already exists
    if [ -f "${output_dir}/${task//,/_}/${SAVE_NAME}" ]; then
        echo "⏭️  SKIP (already exists): ${output_dir}/${task//,/_}/${SAVE_NAME}"
        return 0
    fi

    echo "==============================================="
    echo "CUDA=$CUDA_VISIBLE_DEVICES"
    echo "MODEL=$model"
    echo "RUN_BO_ON=$run_bo_on"
    echo "TASK=$task"
    echo "TRAINING_TASKS=$training_tasks"
    echo "OPT_METHOD=$opt_method"
    echo "ACQ_FUNC=$acq_func"
    echo "EVAL_METHOD=$eval_method"
    echo "JOBS_PREDICTOR_MODEL=$jobs_predictor_model"
    echo "USE_JOBS=$USE_JOBS"
    echo "NUM_INITIAL_RANDOM_SAMPLES=$NUM_INITIAL_RANDOM_SAMPLES"
    echo "OUTPUT AT ${LOG_FILE}"
    echo "RESULTS WILL BE SAVED AT ${output_dir}/${task//,/_}/${SAVE_NAME}"
    echo "==============================================="

    nohup python3 -u BO_runs_LLM_joint_optimization.py \
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
        --jobs_predictor_model=$jobs_predictor_model \
        --ucb_beta=$UCB_BETA \
        --cost_scale_mf=$COST_SCALE_MF \
        --optimize_method=$opt_method \
        --num_initial_random_samples=$NUM_INITIAL_RANDOM_SAMPLES \
        --output_dir=$output_dir \
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

echo "==============================================="
echo "Sweep Configuration"
echo "==============================================="
echo "OPT_METHODS: ${OPT_METHODS[*]}"
echo "ACQ_FUNCS: ${ACQ_FUNCS[*]}"
echo "EVAL_METHODS: ${EVAL_METHODS[*]}"
echo "RUN_BO_ON: ${RUN_BO_ON_OPTIONS[*]}"
echo "MODELS: ${MODELS[*]}"
echo "JOBS_PREDICTOR_MODELS: ${JOBS_PREDICTOR_MODELS[*]}"
echo "TRAINING_TASKS: ${TRAINING_TASKS_OPTIONS[*]}"
echo "EVAL_TASKS: ${TASKS[*]}"
echo "ITERATIONS: $ITER | TRIALS: $TRIALS | SEED: $seed | NUM_INITIAL_RANDOM_SAMPLES: $NUM_INITIAL_RANDOM_SAMPLES"
echo "==============================================="
echo ""

for model in "${MODELS[@]}"; do
    for run_bo_on in "${RUN_BO_ON_OPTIONS[@]}"; do
        for training_tasks in "${TRAINING_TASKS_OPTIONS[@]}"; do
            for task in "${TASKS[@]}"; do
                for opt_method in "${OPT_METHODS[@]}"; do
                    for acq_func in "${ACQ_FUNCS[@]}"; do
                        for eval_method in "${EVAL_METHODS[@]}"; do
                            if [ "$USE_JOBS" -eq 1 ]; then
                                for jobs_predictor_model in "${JOBS_PREDICTOR_MODELS[@]}"; do
                                    run_job "$task" "$opt_method" "$acq_func" "$eval_method" "$run_bo_on" "$model" "$training_tasks" "$jobs_predictor_model"
                                done
                            else
                                run_job "$task" "$opt_method" "$acq_func" "$eval_method" "$run_bo_on" "$model" "$training_tasks" ""
                            fi
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
echo "Summary"
echo "==============================================="

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "🎉 All jobs completed successfully!"
else
    echo "❌ ${#FAILED_JOBS[@]} job(s) failed:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "   - $job"
    done
fi