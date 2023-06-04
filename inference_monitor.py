import pandas as pd
from ast import literal_eval
import numpy as np
import torch
import matplotlib.pyplot as plt
from miditoolkit import MidiFile
import pypianoroll
from fractions import Fraction

from dioai.preprocessor.utils.container import (
    WordSequence, 
    MetaInfo, 
    ChordInfo, 
    GuidelineInfo
)
from dioai.preprocessor.utils.constants import (
    DEFAULT_NUM_BEATS, 
    DEFAULT_TICKS_PER_BEAT, 
    NOTE_RESOLUTION, 
    VELOCITY_BINS
)
from dioai.preprocessor.offset import (
    VOCAB_SIZE, 
    SpecialToken, 
    SequenceWord, 
    ChordWord
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

# generating
with torch.no_grad():
    init_sequence, init_memory = generator.init_input_sequence(
        encoded_input.input_concatenated
    )
    sequence, contexts, probs = generator.generate_sequence(
        sequence=init_sequence, memory=init_memory, context=Context()
    )
sequence = np.array(sequence)
# monitoring
bar_token_positions = np.where(sequence==SequenceWord.BAR.value.offset)[0]
generated_sequence = sequence[bar_token_positions[0]:]
num_bars = len(bar_token_positions)

decoded_midi = decode_midi(
    meta_info=meta_info,
    chord_info=chord_info,
    word_sequence=WordSequence.create(generated_sequence.tolist())
)
DECODED_PATH = WORKING_PATH + "/ex.mid"
decoded_midi.dump(DECODED_PATH)

decoded_midi_obj = pypianoroll.read(DECODED_PATH)
decoded_midi_obj.plot()
decoded_midi_obj.tracks[0].plot()

re_encoded_sequence = NoteSequenceEncoder().encode(MidiFile(DECODED_PATH), sample_info)
generated_sequence == re_encoded_sequence





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

    def visualize(self, ax:plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax = self._draw(ax)
            plt.show()
            plt.close()
        else: 
            return self._draw(ax)


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
    

class TokenMonitor:
    def __init__(self, context:Context, prob:LogitProbs, token):
        self.context = context
        self.prob = prob
        self.selected_token = token

    def interpret(self):
        pass





class SequentialDetector:
    RANGES: list[range]
    def __init__(self):
        self.num_components = len(self.RANGES)
        self.count = -1
        self.initialize()

    def initialize(self):
        self.i = 0
        self.components = []
        self.completion = np.array(self.num_components * [False])

    def detect(self, token):
        self.components.append(token)

        if (token in self.RANGES[self.i] and 
            self.completion[:self.i].all() and not self.completion[self.i:].any()):
            self.completion[self.i] = True
            if self.i == self.num_components-1:
                self.count += 1
                return True
            else:
                self.i += 1
                return None
        else:
            return False

class NoteDetector(SequentialDetector):
    RANGES = [
        SequenceWord.NOTE_POSITION.value.vocab_range,
        SequenceWord.VELOCITY.value.vocab_range,
        SequenceWord.PITCH.value.vocab_range,
        SequenceWord.NOTE_DURATION.value.vocab_range
    ]

    def __init__(self):
        super().__init__()


class ChordDetector(SequentialDetector):
    RANGES = [
        ChordWord.CHORD_POSITION.value.vocab_range,
        ChordWord.CHORD.value.vocab_range,
        ChordWord.CHORD_DURATION.value.vocab_range
    ]

    def __init__(self):
        super().__init__()



class SequenceMonitor:
    BAR_TOKEN = SequenceWord.BAR.value.offset
    EOS_TOKEN = SpecialToken.EOS.value.offset
    CHORD_VOCAB_RANGE = range(ChordWord.CHORD_POSITION.value.offset, 
                              ChordWord.CHORD_DURATION.value.last)

    def __init__( 
            self, 
            meta_info:MetaInfo, 
            chord_info, 
            contexts, 
            probs, 
            sequence:np.ndarray
        ):
        self.meta_info = meta_info
        self.chord_info = chord_info
        self.sequence = sequence

        self._parse_and_group()
        self._transform()

        bar_token_positions = np.where(sequence==SequenceWord.BAR.value.offset)[0]

        # self.sequence = sequence[bar_token_positions[0]:]
        # self.sequence = [
        #     TokenMonitor(contexts[i], LogitProbs(probs[i]), self.sequence[i]) 
        #     for i in range(len(contexts))
        # ]
        
        # self.num_bars = len(bar_token_positions)
        # self.piano_roll = self.num_bars * [np.zeros((
        #     SequenceWord.PITCH.value.vocab_size, 
        #     SequenceWord.NOTE_DURATION.value.vocab_size))]
        # self._record_to_piano_roll()

    def _parse_and_group(self):
        note_detector = NoteDetector()
        chord_detector = ChordDetector()

        def feed_token_to_detector(detector:SequentialDetector, token):
            detection_result = detector.detect(token)
            if detection_result is True:
                self.parsed_items.append(detector.components)
            elif detection_result is False:
                self.invalid_tokens.extend(detector.components)
            else:
                return
            detector.initialize()
        
        self.parsed_items = []
        self.invalid_tokens = []
        for token in self.sequence:
            if token == self.BAR_TOKEN or token == self.EOS_TOKEN:
                self.parsed_items.append([token])
            elif token in self.CHORD_VOCAB_RANGE:
                feed_token_to_detector(chord_detector, token)
            else:
                feed_token_to_detector(note_detector, token)

    def _transform(self):
        pass

    def _record_to_piano_roll(self):
        pass

    def visualize(self, visual_midi=True):
        pass


hist = SequenceMonitor(
    contexts=contexts, 
    probs=probs, 
    sequence=generated_sequence,
    meta_info=meta_info,
    chord_info=chord_info
)


hist._parse_and_group()
hist.parsed_items

hist.sequence[7].prob.visualize()


