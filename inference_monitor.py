import pandas as pd
from pprint import pprint

WORKING_PATH = "/home/honggyu/workspace"
VOCAB_SIZE = 956

df = pd.read_csv("/home/honggyu/workspace/_datasets/raw_data/2023-05-12/sample_info.csv")
pprint(df.iloc[2400].to_dict())

generator_args = {
    # Model Arguments
    "checkpoint_dir": WORKING_PATH + "/_model_checkpoints/20230516-115949",
    # Input Arguments
    "output_dir": WORKING_PATH + "/_outputs",
    # Input meta
    "bpm": 140,
    "audio_key": "cminor",
    "time_signature": "3/4",
    "pitch_range": "mid_high",
    "num_measures": 8,
    "is_incomplete_measure": False,
    "inst": "acoustic_piano",
    "genre": "jazz",
    "track_category": "main_melody",
    "rhythm": "swing",
    "min_velocity": 2,
    "max_velocity": 127,
    # Guideline info
    "figure_a": "솔로잉",
    "pitch_a": "단선율",
    "duration_b": "8분음표",
    # Inference 시 필요 정보
    "num_generate": 2,
    "top_k": 32,
    "temperature": 0.95,
    # 생성에 사용할 코드 진행을 가져올 샘플 ID
    "chord_sample_id": '1261260081003-83-012',
}

import torch
import numpy as np
import matplotlib.pyplot as plt
from dioai.transformer_xl.midi_generator.inferrer import InferenceTask
from dioai.transformer_xl.midi_generator.model_initializer import ModelInitializeTask
from dioai.transformer_xl.midi_generator.generate_pipeline import PreprocessTask
from dioai.transformer_xl.midi_generator.inference.context import Context
from dioai.preprocessor.offset import SpecialToken, SequenceWord, ChordWord, MetaWord, GuidelineWord

preprocessor = PreprocessTask()
model_initializer = ModelInitializeTask(map_location='cuda', device=torch.device('cuda'))
generator = InferenceTask(device='cuda')

encoded_input = preprocessor.execute(generator_args)
model = model_initializer.execute()
generator(
    model=model,
    input_data=preprocessor.input_data,
    inference_cfg=model_initializer.inference_cfg
)

class TokenInterface:
    def __init__(self):
        pass

    def visualize_prob(self):
        pass



def token_interpreter(token):

    return token

with torch.no_grad():
    init_sequence, init_memory = generator.init_input_sequence(
        encoded_input.input_concatenated
    )
    sequence, memory, context = init_sequence, init_memory, Context()
    sequence, memory, context, probabilities = generator.process(sequence, memory, context=context)
    print(token_interpreter(sequence[-1]))
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0,VOCAB_SIZE), probabilities.cpu().detach().numpy())
    plt.show()
