import pandas as pd
import numpy as np
from pprint import pprint
from ast import literal_eval
import torch
import matplotlib.pyplot as plt
from dioai.preprocessor.utils.container import MetaInfo
from dioai.transformer_xl.midi_generator.inferrer import InferenceTask
from dioai.transformer_xl.midi_generator.model_initializer import ModelInitializeTask
from dioai.transformer_xl.midi_generator.generate_pipeline import PreprocessTask
from dioai.transformer_xl.midi_generator.inference.context import Context
from dioai.preprocessor.offset import VOCAB_SIZE, SpecialToken, SequenceWord, ChordWord
from dioai.preprocessor.decoder.midi_writer import NoteWriter
from dioai.preprocessor.utils.container import WordSequence

WORKING_PATH = "/home/honggyu/workspace"
SAMPLE_INFO_PATH = WORKING_PATH + "/_datasets/raw_data/2023-05-12/sample_info.csv"
GUDELINE_KEYS = (
    "figure_a", "pitch_a", "duration_b", "position_b", "position_c", "position_e", "position_f", "position_g"
)
df = pd.read_csv(SAMPLE_INFO_PATH)


sample_info = df.iloc[2000].to_dict()

sample_info.update({"rhythm": sample_info["sample_rhythm"]})
sample_info.update({"audio_key": sample_info["audio_key"] + sample_info["chord_type"]})

meta_info = MetaInfo(**sample_info)
pprint(vars(meta_info))

generator_args = {}
generator_args.update({
    "checkpoint_dir": WORKING_PATH + "/_model_checkpoints/20230516-115949",
    "output_dir": WORKING_PATH + "/_outputs",
})
generator_args.update(vars(meta_info))
generator_args.update({"chord_progression": literal_eval(sample_info["chord_progressions"])[0]})
# generator_args.update({"chord_sample_id": sample_info["id"]})
for key in GUDELINE_KEYS: generator_args.update({key: sample_info[key]})
generator_args.update({"num_generate": 3, "top_k": 32, "temperature": 0.95})


preprocessor = PreprocessTask()
encoded_input = preprocessor.execute(generator_args)

model_initializer = ModelInitializeTask(map_location='cuda', device=torch.device('cuda'))
generator = InferenceTask(device='cuda')
generator(
    model=model_initializer.execute(),
    input_data=preprocessor.input_data,
    inference_cfg=model_initializer.inference_cfg
)

class PositionProb:
    pass

class VelocityProb:
    pass

class DurationProb:
    pass

class PitchProb:
    def __init__(self, pitch_prob, current_chord):
        self.prob = pitch_prob
        self.chord = current_chord

    def _draw(self):
        pass

    def visualize(self):
        fig, ax = plt.subplots()
        ax = self._draw(ax)
        plt.show()
        plt.close()

class LogitProbs:
    position_range = SequenceWord.NOTE_POSITION.value.vocab_range
    velocity_range = SequenceWord.VELOCITY.value.vocab_range
    pitch_range = SequenceWord.PITCH.value.vocab_range
    duration_range = SequenceWord.NOTE_DURATION.value.vocab_range
    chord_position_range = ChordWord.CHORD_POSITION.value.vocab_range
    chord_range = ChordWord.CHORD.value.vocab_range
    chord_duration_range = ChordWord.CHORD_DURATION.value.vocab_range

    def __init__(self, prob):
        self.prob = np.array(prob.detach().cpu().numpy())

    @property
    def special(self):
        return self.prob[:2]
        
    @property
    def position(self):
        return PositionProb(self.prob[self.position_range])
    
    @property
    def velocity(self):
        return VelocityProb(self.prob[self.velocity_range])
    
    @property
    def pitch(self):
        return PitchProb(self.prob[self.pitch_range])
    
    @property
    def duration(self):
        return DurationProb(self.prob[self.duration_range])
    
    @property
    def chord_position(self):
        return self.prob[self.chord_position_range]
    
    @property
    def chord(self):
        return self.prob[self.chord_range]
    
    @property
    def chord_duration(self):
        return self.prob[self.chord_duration_range]
    
    def visualize(self):
        fig, axes = plt.subplots(4,2, figsize=(10,10))
        ax:plt.Axes = axes[0,1]
        ax.bar(self.position_range, self.position)
        ax.set_ylim([0,1])
        ax.set_yscale('symlog')
        ax.set_title('Position')
        
        ax = axes[1,1]
        ax.bar(self.velocity_range, self.velocity)
        ax.set_ylim([0,1])
        ax.set_yscale('symlog')
        ax.set_title('Velocity')

        ax = axes[2,1]
        ax.bar(self.pitch_range, self.pitch)
        ax.set_ylim([0,1])
        ax.set_yscale('symlog')
        ax.set_title('Pitch')

        ax = axes[3,1]
        ax.bar(self.duration_range, self.duration)
        ax.set_ylim([0,1])
        ax.set_yscale('symlog')
        ax.set_title('Duration')

        fig.tight_layout()
        plt.show()
        plt.close()
    

class GeneratedToken:
    def __init__(self, context:Context, prob:LogitProbs, token):
        self.context = context
        self.prob = prob
        self.selected_token = token

    def interpret(self):
        pass

class GenerationHistory:
    def __init__(self, sequence, contexts, probs):
        self.hist = [GeneratedToken(contexts[i], LogitProbs(probs[i]), sequence[i]) 
                     for i in range(len(contexts))]



# generator.execute(encoded_input)

with torch.no_grad():
    init_sequence, init_memory = generator.init_input_sequence(
        encoded_input.input_concatenated
    )
    sequence, contexts, probs = generator.generate_sequence(
        sequence=init_sequence, memory=init_memory, context=Context()
    )
    generation_hist = GenerationHistory(contexts=contexts, probs=probs, sequence=sequence)

generation_hist.hist[7].prob.visualize()

prob = generation_hist.hist[202].prob
fit, ax = plt.subplots()
ax.bar(prob.pitch_range, prob.pitch)


class SequenceViewer:
    position_range = SequenceWord.NOTE_POSITION.value.vocab_range
    velocity_range = SequenceWord.VELOCITY.value.vocab_range
    pitch_range = SequenceWord.PITCH.value.vocab_range
    NUM_PITCHES = SequenceWord.PITCH.value.vocab_size
    duration_range = SequenceWord.NOTE_DURATION.value.vocab_range
    NUM_RESOLUTIONS = SequenceWord.NOTE_DURATION.value.vocab_size
    chord_position_range = ChordWord.CHORD_POSITION.value.vocab_range
    chord_range = ChordWord.CHORD.value.vocab_range
    chord_duration_range = ChordWord.CHORD_DURATION.value.vocab_range
    decoder = NoteWriter()

    def __init__(self, sequence):
        bar_token_positions = np.where(np.array(sequence)==SequenceWord.BAR.value.offset)[0]
        self.num_bars = len(bar_token_positions)
        self.note_sequence = sequence[bar_token_positions[0]:]
        self.proll = self.num_bars * [np.zeros((
            SequenceWord.PITCH.value.vocab_size, 
            SequenceWord.NOTE_DURATION.value.vocab_size))]
        self._record()

    def _record(self):
        bar_idx = -1
        for token in self.sequence:
            if token == 2:
                bar_idx += bar_idx
                continue
            if token in self.position_range:
                self.proll[bar_idx][self.NUM_PITCHES-pitch,]

    def visualize(self):
        pass

viewer = SequenceViewer(sequence)
viewer.num_bars

bar_token_positions = np.where(np.array(sequence)==SequenceWord.BAR.value.offset)[0]
num_bars = len(bar_token_positions)
note_sequence = sequence[bar_token_positions[0]:]
sequence_info = {}
sequence_info.update(vars(meta_info))
sequence_info.update({"chord_progression": generator_args["chord_progression"]})
sequence_info.update({"note_sequence": note_sequence})
decoder = NoteWriter(
    meta_info=meta_info, 
    word_sequence=WordSequence.create(**sequence_info)
)