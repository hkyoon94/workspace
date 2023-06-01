from preprocess import main as preprocess_main

WORKING_PATH = "/home/honggyu/workspace"

class PreprocessMainArgs:
    root_dir = WORKING_PATH + "/_datasets/raw_data"
    output_dir = WORKING_PATH + "/_datasets/processed_data/"
    num_cores = 8
    steps = ("split", "encode", "augment")

preprocess_main(PreprocessMainArgs())

 
# ----------------------------------------------------------------------------------
from train_xl import main as train_xl_main

class TrainerArgs:
    data_dir = WORKING_PATH + "/_datasets/processed_data/"
    work_dir = WORKING_PATH + "/_model_checkpoints"
    training_cfg = (
        WORKING_PATH
        + "/dio_ai/dioai/transformer_xl/\
        training_config_xl/experiment_baseline.yml"
    )
    val = True
    model = "xl"

train_xl_main(TrainerArgs())


# ----------------------------------------------------------------------------------
from inference_xl import main as inference_xl_main
from itertools import product

generator_args = {
    # Model Arguments
    "checkpoint_dir": (WORKING_PATH + "/_model_checkpoints/20230516-115949",),
    # Input Arguments
    "output_dir": (WORKING_PATH + "/_outputs",),
    # Input meta
    "bpm": (140,),
    "audio_key": ("cminor", "fmajor"),
    "time_signature": ("3/4", "4/4"),
    "pitch_range": ("low", "mid", "mid_high"),
    "num_measures": (8,),
    "is_incomplete_measure": (False,),
    "inst": ("acoustic_piano", "string_violin", "acoustic_bass"),
    "genre": ("jazz", "cinematic"),
    "track_category": ("main_melody", "accompaniment", "bass"),
    "rhythm": ("standard", "swing"),
    "min_velocity": (2,),
    "max_velocity": (127,),
    # Guideline info
    "figure_a": ("솔로잉", "컴핑", "아르페지오"),
    "pitch_a": ("단선율", "3성부"),
    "duration_b": ("4분음표", "8분음표"),
    # Inference 시 필요 정보
    "num_generate": (2,),
    "top_k": (32,),
    "temperature": (0.95,),
    # 생성에 사용할 코드 진행을 가져올 샘플 ID
    "chord_sample_id": ("1290530061002-80-012",),
}

# Get the keys and values from the dictionary
keys = list(generator_args.keys())
values = list(generator_args.values())

generator_args_ = []

# Iterate through every combination
for combination in product(*values):
    combination_dict = dict(zip(keys, combination))
    generator_args_.append(combination_dict)

total_num = len(generator_args)
for i, args in enumerate(generator_args_):
    inference_xl_main(args)
    print(f"Generated: {i}/{total_num}")