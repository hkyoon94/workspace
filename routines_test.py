from fractions import Fraction

import numpy as np

from dioai.preprocessor.encoder.chord.container import (
    ChordDetector,
    ChordName,
    ChordTokenComponents,
)

beats_per_bar = Fraction("4/4") * 4

parser = ChordDetector(
    chord_progression=[
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Bsus4",
        "Bsus4",
        "Bsus4",
        "Bsus4",
        "D",
        "D",
        "D",
        "D",
    ],
    beats_per_bar=beats_per_bar,
)

parser._chords_per_bar
parser._num_measures
parser._split_by_bar
parser.chord_indices
parser.chord_names
parser.chord_durations

parser = ChordTokenComponents(
    chord_progression=[
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Am7b5",
        "Bsus4",
        "Bsus4",
        "Bsus4",
        "Bsus4",
        "D",
        "D",
        "D",
        "D",
    ],
    time_signature="4/4",
)
parser.chord_token
parser.chord_position
parser.chord_duration

chord = ChordName(chord="C#mM7(#11#13b9)/F")
chord.root
chord.quality
chord.tension
chord.bass


# --------- pydantic 사용 예시

from pydantic import BaseModel, validator

class SubSubClass(BaseModel):
    a: int

class SubClass1(BaseModel):
    a1: SubSubClass
    a2: int

class SubClass2(BaseModel):
    b1: int
    b2: str

    @validator("b2")
    def validate_b2(cls, v):
        if v != "5":
            raise ValueError("b2 is not '5'")
        return v

class SuperClass(BaseModel):
    sub_1: SubClass1
    sub_2: SubClass2

    @classmethod
    def create(cls, **kwargs):
        return cls(sub_1=kwargs, sub_2=kwargs)


input_dict = {"a1": 2, "a2": 3, "b1": 4, "b2": 5}


super_class = SuperClass.create(**input_dict)
# super_class = SuperClass(**input_dict)
super_class.sub_1.a1
super_class.sub_1.a2
super_class.sub_2.b1
super_class.sub_2.b2

input_dict = {
    "sub_1": {"a1": {"a": 15, "b": 30}, "a2": 2},
    "sub_2": {"b1": 3, "b2": 5 },
    "sub_3": None
}
super_class = SuperClass(**input_dict)

super_class.sub_2.validate_b2

sub_class = SubClass2(**input_dict)
sub_class.b1
sub_class.b2

# getattr(SubClass2,'b2')


# --------- decorator 사용 예시

import functools


def decorate_1(deco2_func):
    @functools.wraps(deco2_func)
    def wrapper(*args, **kwargs):
        # pre-call
        print(deco2_func.__name__, "함수 시작")
        # main-call
        value = deco2_func(*args, **kwargs)
        # after-call
        print(deco2_func.__name__, "함수 끝")
        return value

    return wrapper


def decorate_2(deco1_func):
    @functools.wraps(deco1_func)
    def wrapper(*args, **kwargs):
        print("second deco")
        value = deco1_func(*args, **kwargs)
        return value

    return wrapper


def decorate_3(arg):
    def decorator(raw_func):
        @functools.wraps(raw_func)
        def wrapper(*args, **kwargs):
            print("third deco")
            value = raw_func(*args, **kwargs) + arg
            print(f"added {arg}, finished calculation!")
            return value

        return wrapper

    return decorator


def plain_deco(func):
    print("This modifies nothing to the func")
    return func


@decorate_1
@decorate_2
@decorate_3(arg=3)
@plain_deco
def add(a, b):
    return a + b


add(2, 3)

add.__name__

c = add(2, 3)
c


# ---------- enum 사용 예시

import enum


class SubStruct(BaseModel):
    a: int


class Struct(enum.Enum):
    A = SubStruct(a=10)
    B = SubStruct(a=20)


getattr(Struct, "A").value.a
getattr(Struct, "A").value


# ---------- token monitor

import numpy as np

input_train = np.load(
    "/home/honggyu/workspace/_datasets/processed_data/output_npy/input_train.npy", allow_pickle=True
)

len(input_train[58])


# -------- using meta encoder

from dioai.preprocessor.encoder.meta.encoder import MetaEncoder
from dioai.preprocessor.utils.container import MetaInfo

meta_dict = {
    "bpm": 70,
    "audio_key": "c#minor",
    "time_signature": "6/8",
    "pitch_range": "mid_high",
    "num_measures": 8,
    "is_incomplete_measure": False,
    "inst": "acoustic_piano",
    "genre": "funk",
    "track_category": "riff",
    "rhythm": "standard",
    "min_velocity": 2,
    "max_velocity": 127,
}

meta_info = MetaInfo(**meta_dict)
MetaEncoder().encode(meta_info)


from dioai.preprocessor.encoder.guideline.encoder import GuidelineEncoder
from dioai.preprocessor.utils.container import GuidelineInfo

guideline_dict = {}


from dioai.transformer_xl.dataset import MusicDataset

dataset = MusicDataset(
    data_dir="/home/honggyu/workspace/_datasets/processed_data/output_npy", cfg=""
)
