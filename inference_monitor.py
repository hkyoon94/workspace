import torch
import json
import pypianoroll
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from miditoolkit import MidiFile
from fractions import Fraction
from ast import literal_eval
from copy import deepcopy
from pprint import pprint as pp
from importlib import reload

from dioai.preprocessor.utils.container import (
    WordSequence, 
    MetaInfo, 
    ChordInfo, 
    GuidelineInfo
)
from dioai.preprocessor.offset import (
    SpecialToken, 
    SequenceWord, 
)
from dioai.preprocessor.encoder.note_sequence.encoder import NoteSequenceEncoder
from dioai.preprocessor.decoder.decoder import decode_midi

from dioai.transformer_xl.midi_generator.model_initializer import ModelInitializeTask
from dioai.transformer_xl.midi_generator.generate_pipeline import PreprocessTask
from dioai.transformer_xl.midi_generator.inferrer import InferenceTask
from dioai.transformer_xl.midi_generator.inference.context import Context

WORKING_PATH = "/home/honggyu/workspace"
SAMPLE_INFO_PATH = WORKING_PATH + "/_datasets/raw_data/2023-05-12/sample_info.csv"
GUDELINE_KEYS = (
    "figure_a", 
    "pitch_a", 
    "duration_b", 
    "position_b", 
    "position_c", 
    "position_e", 
    "position_f", 
    "position_g"
)
df = pd.read_csv(SAMPLE_INFO_PATH, converters={"chord_progressions": literal_eval})
df.rename(columns={"sample_rhythm": "rhythm"}, inplace=True)
df["audio_key"] = df["audio_key"] + df["chord_type"]


# building metadata for generating
sample_info = df.iloc[2000].to_dict()

meta_info = MetaInfo(**sample_info)
guideline_info = GuidelineInfo.create(**sample_info)
chord_info = ChordInfo(
    **sample_info, 
    chord_progression=sample_info["chord_progressions"][0],
    # chord_progression=[
    #     'G', 'G', 'G', 'G', ...
    # ]
)

# preparing generator
generator_args = {}
generator_args.update({
    "checkpoint_dir": WORKING_PATH + "/_model_checkpoints/20230516-115949",
    "output_dir": WORKING_PATH + "/_outputs",
    "num_generate": 3, 
    "top_k": 32, 
    "temperature": 0.95
})
generator_args.update(vars(meta_info))

for key in GUDELINE_KEYS: 
    generator_args.update({key: sample_info[key]})

generator_args.update(vars(chord_info))
# generator_args.update({"chord_sample_id": sample_info["id"]})


# initializing generator
model_initializer = ModelInitializeTask(map_location='cuda', device=torch.device('cuda'))
model = model_initializer.execute()

preprocessor = PreprocessTask()
encoded_input = preprocessor.execute(generator_args)

generator = InferenceTask(device='cuda')
generator(
    model=model, 
    input_data=preprocessor.input_data,
    inference_cfg=model_initializer.inference_cfg
)

# GENERATING TOKENS
with torch.no_grad():
    init_sequence, init_memory = generator.init_input_sequence(
        encoded_input.input_concatenated
    )
    sequence_pre, memory, context = deepcopy(init_sequence), init_memory, Context()
    probs, contexts, ignored_probs, ignored_contexts, ignored_iters = [], [], [], [], []

    for i in range(generator.inference_cfg.GENERATION.generation_length):
        contexts.append(context)

        sequence_new, memory, context, prob = generator.process(deepcopy(sequence_pre), memory, context)
        probs.append(deepcopy(prob))

        if len(sequence_pre) + 1 != len(sequence_new):
            print(f"WARNING: Token ignored on iteration {i}!!!\n\t(Context type: {context.types})")
            ignored_contexts.append(contexts.pop())
            ignored_probs.append(probs.pop())
            ignored_iters.append(i)

        if sequence_new[-1] == SpecialToken.EOS.value.offset:
            break

        sequence_pre = deepcopy(sequence_new)
    sequence_final = np.array(sequence_new)


# MONITORING

import dioai.monitor.sequence_monitor as sm
import dioai.monitor.unions as unions
import dioai.monitor.tokens as tokens
import dioai.monitor.logitprobs as logitprobs
reload(logitprobs)
reload(tokens)
reload(unions)
reload(sm)

from dioai.monitor.sequence_monitor import SequenceMonitor
from dioai.monitor.logitprobs import PitchProb
from time import time

st = time()
hist = SequenceMonitor(meta_info, chord_info, sequence_final, contexts, probs)
ft = time(); print(f"{ft-st:.4f}")

print(hist.chord_unions[1])
print(hist.bar_unions[0])
print(hist.bar_unions[1].group)
print(hist.bar_unions[2].group)
print(len(hist.chord_unions))
print(len(hist.bar_unions))

union:unions.HarmonicUnion = hist.chord_unions[0]
prob = PitchProb(union.notes[0].tokens[2].prob)
prob.pitch.to_matrix
len(prob.prob_filtered)
prob.visualize()


union.group
union.chord
union.group

decoded_midi = decode_midi(
    meta_info=meta_info,
    chord_info=chord_info,
    word_sequence=WordSequence.create(sequence_final.tolist())
)
DECODED_PATH = WORKING_PATH + "/ex.mid"
decoded_midi.dump(DECODED_PATH)

decoded_midi_obj = pypianoroll.read(DECODED_PATH)
decoded_midi_obj.plot()
decoded_midi_obj.tracks[0].plot()

re_encoded_sequence = NoteSequenceEncoder().encode(MidiFile(DECODED_PATH), sample_info)
