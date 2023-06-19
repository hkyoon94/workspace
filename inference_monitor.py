import torch
import pandas as pd
import numpy as np
from ast import literal_eval
from copy import deepcopy
from importlib import reload
from time import time

from dio_ai.dioai.preprocessor.utils.container import MetaInfo, GuidelineInfo, ChordInfo
from dio_ai.dioai.preprocessor.offset import SpecialToken
from dio_ai.dioai.preprocessor.encoder.note_sequence.encoder import NoteSequenceEncoder

from dio_ai.dioai.transformer_xl.midi_generator.model_initializer import ModelInitializeTask
from dio_ai.dioai.transformer_xl.midi_generator.generate_pipeline import PreprocessTask
from dio_ai.dioai.transformer_xl.midi_generator.inferrer import InferenceTask
from dio_ai.dioai.transformer_xl.midi_generator.inference.context import Context


WORKING_PATH = "/home/honggyu/workspace/dio_ai"
SAMPLE_INFO_PATH = WORKING_PATH + "/data/raw_data/2023-06-07/sample_info.csv"
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
sample_info = df.iloc[27000].to_dict()
# sample_info.update({'track_category': ''})
sample_info

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
    "checkpoint_dir": WORKING_PATH + "data/XL/model_ckpt",
    "output_dir": WORKING_PATH + "data/XL/outputs",
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
generator(model=model, 
          input_data=preprocessor.input_data,
          inference_cfg=model_initializer.inference_cfg)

# sequences = generator.execute(encoded_input=encoded_input)


# GENERATING TOKENS
with torch.no_grad():
    init_sequence, init_memory = generator.init_input_sequence(
        encoded_input.input_concatenated
    )
    sequence_pre, memory, context = init_sequence[:], init_memory[:], Context()
    logits, contexts, ignored_logits, ignored_contexts, ignored_iters = [], [], [], [], []

    for i in range(generator.inference_cfg.GENERATION.generation_length):
        contexts.append(context)

        sequence_new, memory, context, logit = generator.process(sequence_pre[:], memory, context)
        logits.append(deepcopy(logit))

        if len(sequence_pre) + 1 != len(sequence_new):
            print(f"WARNING: Token ignored on iteration {i+1}!!!\n\t(Context type: {context.types})")
            ignored_contexts.append(contexts.pop())
            ignored_logits.append(logits.pop())
            ignored_iters.append(i)

        if sequence_new[-1] == SpecialToken.EOS.value.offset:
            print(f"Finished generating: {len(sequence_new) - len(init_sequence)} tokens / iteration: {i+1}")
            break

        sequence_pre = sequence_new[:]
    sequence_final = np.array(sequence_new)


# MONITORING

import dio_ai.dioai.sequence_analyzer.exceptions as exceptions
import dio_ai.dioai.sequence_analyzer.logitprobs as logitprobs
import dio_ai.dioai.sequence_analyzer.token as token
import dio_ai.dioai.sequence_analyzer.items as items
import dio_ai.dioai.sequence_analyzer.unions as unions
import dio_ai.dioai.sequence_analyzer.sequence as sequence
reload(exceptions)
reload(logitprobs)
reload(token)
reload(items)
reload(unions)
reload(sequence)

from dio_ai.dioai.sequence_analyzer.sequence import ParsedSequence

seq = ParsedSequence(
    meta_info=meta_info, 
    chord_info=chord_info, 
    sequence=sequence_final, 
    contexts=contexts, 
    logits=logits,
)
seq.get_unions()

seq.play()
seq.meta_info

seq.parsed_items
seq.harmonic_unions[6]
seq.harmonic_unions[0].plot()
seq.harmonic_unions[1].notes[0].tokens[2].logit.pitch.plot()
seq.harmonic_unions[1].notes[0].tokens[2].logit.probs.pitch.plot()
seq.harmonic_unions[1].notes[0].tokens[0].logit.softmax().top_k().note_position.plot()

seq.harmonic_unions[0]
seq.harmonic_unions[0].plot()
seq.harmonic_unions[0].notes[1].plot_logit()

union = seq.bar_unions[0]
union.onsets()


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dio_ai.dioai.sequence_analyzer.items import (
    compute_tonal_centroid, compute_chord_chroma, compute_tonal_distance, NUM_CHORD_QUALITY
)

diatonic_qualities = np.array([0,3,3,0,0,3,6])
scales = NUM_CHORD_QUALITY * np.array([0,2,4,5,7,9,11])
diatonic_chords_ = diatonic_qualities + scales
diatonic_chords = diatonic_chords_

labels = 7 * [0]
for i in range(1,12):
    labels += 7 * [i]
    diatonic_chords = np.concatenate((diatonic_chords, diatonic_chords_+i))

tonal_centroids = np.zeros((len(labels),6))
for i, token in enumerate(diatonic_chords):
    tonal_centroids[i] = compute_tonal_centroid(compute_chord_chroma(token))

res = TSNE(n_components=2).fit_transform(tonal_centroids)

fig, ax = plt.subplots()
ax.scatter(res[:,0], res[:,1], c=labels)
plt.show()
plt.close()

compute_tonal_distance(0,99) # C, B
compute_tonal_distance(12,13)
