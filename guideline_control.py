import pandas as pd
from ast import literal_eval
from glob import glob

from dio_ai.dioai.preprocessor.utils.container import MetaInfo, GuidelineInfo, ChordInfo
from dioai.classification.classify_sample import GuidelineClassifier
from dioai.classification.classifier import ClassifierPredictor


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

midi_id = sample_info.get('id')
midi_dir = "/home/honggyu/workspace/dio_ai/data/raw_data/2023-06-07/trimmed"
midi_path = glob(f"{midi_dir}/{midi_id}*.mid")
sample_info

classifier = GuidelineClassifier('figure_a')
classifier.classify(sample_info, value=None)