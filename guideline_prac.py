import pandas as pd
import numpy as np
import pickle
from ast import literal_eval
from dioai.utils import load_guideline
from glob import glob 

sample_paths = glob('/home/honggyu/workspace/_datasets/raw_data/2023-05-12/raw/*')

# integrated_sample_infos = pd.read_csv("/home/honggyu/workspace/_datasets/raw_data/2023-04-26/sample_info.csv")
integrated_infos = pd.read_csv(
    "/home/honggyu/workspace/_datasets/raw_data/2023-05-12/sample_info.csv",
    converters={"chord_progressions": literal_eval, "customers": literal_eval}
)

i_ids = integrated_infos['id']
i_ids = i_ids.to_numpy()



# m_ids = meta_infos['id']
# m_ids = m_ids.to_numpy()

# tf_arr = [True] * len(m_ids)
# unintegrable_ids = []
# for i, i_sample in enumerate(m_ids):
#     if i_sample not in i_ids:
#         tf_arr[i] = False
#         unintegrable_ids.append(m_ids[i])

# len(tf_arr)
# len(unintegrable_ids)

from pprint import pprint

import pandas as pd
from miditoolkit import MidiFile
from dioai.preprocessor.parser.meta import MetaParser, remove_number_from_inst
from dioai.preprocessor.encoder.meta.encoder import MetaEncoder
from dioai.preprocessor.encoder.note_sequence.encoder import NoteSequenceEncoder

meta_infos = pd.read_csv(
    "/home/honggyu/workspace/_datasets/raw_data/2023-05-12/meta_info.csv",
    dtype=object
)
# display(meta_infos)

sample_id = 20

sample_meta = meta_infos.loc[sample_id].to_dict()
sample_meta['inst'] = remove_number_from_inst(sample_meta['inst'])

sample_midi = MidiFile(f"/home/honggyu/workspace/_datasets/raw_data/2023-05-12/raw/{sample_meta['id']}.mid")

pprint(sample_meta)

parsed_meta = MetaParser().parse(sample_dict=sample_meta, midi_obj=sample_midi)
tokenized_meta = MetaEncoder().encode(parsed_meta)

print(tokenized_meta)

tokenized_note_sequence = NoteSequenceEncoder().encode(sample_info=sample_meta, midi_obj=sample_midi)

tokenized_note_sequence

x_single_sample = np.array(tokenized_meta + tokenized_note_sequence.tolist())

x_single_sample


# for i, id in enumerate(unintegrable_ids):
#     row = meta_infos[meta_infos['id'] == unintegrable_ids[i]]



#### GUIDELINE INTEGRATION
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from pprint import pprint

from dioai.utils import load_guideline, encode_song_form
from dioai.preprocessor.preparing_steps.integrate_guideline import filter_guideline, split_by_guide
from dioai.preprocessor.guideline.container import GuideID, RawGuideline, RawSamples, song_form_operator
from dioai.preprocessor.guideline.control import Control
from dioai.preprocessor.guideline.unclassified_guideline import UnclassifiedGuideLine
from dioai.preprocessor.guideline.classified_guideline import ClassifiedGuideLine
from dioai.preprocessor.guideline.classifier import Classifiers
from dioai.preprocessor.guideline.constants import (
    AND,
    ANDOR,
    KEYWORD_COLUMN_VARIABLE, 
    PREFERENCE_AND_OR,
    PREFERENCE_COLUMN_VARIABLE,
    SINGLE,
)
from dioai.preprocessor.utils import UNKNOWN

meta_path = "/home/honggyu/workspace/_datasets/raw_data/2023-05-12/meta_info.csv"
guideline_path = "/home/honggyu/workspace/_datasets/raw_data/2023-05-12/guideline_info.csv"
error_id_path = "/home/honggyu/workspace/_datasets/raw_data/2023-05-12/error_info.csv"


meta_info = pd.read_csv(meta_path, dtype=object)
meta_info["encoded_song_form"] = meta_info["form_names"].apply(encode_song_form)

guideline_info = load_guideline(filter_guideline(pd.read_csv(guideline_path,dtype=object)))
guideline_info = split_by_guide(guideline_info)

error_info = pd.read_csv(error_id_path).set_index(keys=["id"])
error_id_list = error_info.index.tolist()


sample_infos_chunk = np.array_split(np.array(guideline_info.to_dict("records")), 50)
sample_infos_chunk = [x.tolist() for x in sample_infos_chunk]


# for i in range(0,100):
split_guideline_info = sample_infos_chunk[1]

for i, guideline_series in enumerate(split_guideline_info):

    i, guideline_series = 7, split_guideline_info[30]
    samples = []

    # for i, guideline_series in tqdm(enumerate(split_guideline_info), desc="integrate"):
    raw_guideline = RawGuideline(guideline=guideline_series)

    # ! Creating UnclassifiedGuideline struct from RawGuideline(dict)
    # unclassified_guideline = UnclassifiedGuideLine.create(raw_guideline)

    component_dict = dict({})
    for name in ("figure", "pitch", "duration", "position", "etc"):
        # name = "figure"
        column_pattern = re.compile(KEYWORD_COLUMN_VARIABLE.get(name))
        # ! column_pattern: '음형', '음고', '음가', '노트 타이밍', '기타'
        # print(column_pattern)

        components: dict[str, dict] = dict()

        for column, value in raw_guideline.guideline.items():
            # print(column, value)
            alphabet = column_pattern.sub("", column)[0]
            # ! column 에 column_pattern과 일치하는 부분이 있으면 ""로 치환한 후, 첫 글자를 리턴
            # ! ex)
            # ! column = '음형A-main_melody', column_pattern = '음형'
            # ! alphabet = 'A'
            # print(alphabet)

            if column_pattern.search(column) and alphabet.isalpha() and value is not None:
                '음형A_'
                field_name = column_pattern.sub(f"{name}_", column).lower().split("-")[0]
                print(field_name)
                components[field_name] = {"keyword": value}

            if any([column.startswith(x) for x in PREFERENCE_AND_OR]):
                preference_name, _ = column.split("-")

                for guideline_field, guideline_value in components.items():
                    if preference_name in str(guideline_value.values()):
                        if value is None:
                            value = ["unknown"]
                        guideline_preference_value = {preference_name: value}
                        components[guideline_field] = guideline_value | guideline_preference_value

        #print(components)
        if name == "pitch":
            component_dict.update({f'{name}es': components})
        else:
            component_dict.update({f'{name}s': components})

    pprint(component_dict)

    unclassified_guideline = UnclassifiedGuideLine(**component_dict)
    
    control = Control.create(raw_guideline).to_dict()

    # ! Creating RawSample for each guide ID
    # raw_samples = RawSamples.create(meta_info, guide_id, error_id_list)

    guide_id = GuideID(guide_id=guideline_series.get("guide_id")[0])

    template_code = guide_id.template_code  # noqa: F841
    track_category = guide_id.track_category  # noqa: F841
    inst = guide_id.inst  # noqa: F841
    song_form = guide_id.song_form

    sample_info = meta_info.query("template_code == @template_code")
    sample_info = sample_info.query("track_category == @track_category")
    sample_info = sample_info.query("inst == @inst")
    if song_form is not None:
        sample_info = sample_info.copy()
        sample_info["song_form_subset"] = sample_info["encoded_song_form"].apply(
            song_form_operator, args=(song_form,)
        )
        song_form_query = (
            "song_form_subset == encoded_song_form | encoded_song_form == 'form127'"
        )
        sample_info = sample_info.query(song_form_query)
    sample_info = sample_info.query("id not in @error_id_list")

    raw_samples = RawSamples(samples=sample_info)

    split = guideline_series.get("split")

    for sample in raw_samples.samples.iloc:
        sample = raw_samples.samples.iloc[0]

        sample["split"] = split
        classified_guideline = ClassifiedGuideLine.create(sample, unclassified_guideline).to_dict()

        name = 'figures'
        name = 'pitches'
        name = 'durations'
        name = 'positions'
        name = 'etcs'

        """category (유형): figures, pitches, durations, positions, etcs"""
        category = getattr(unclassified_guideline, name)
        # ex) category = UnclassifiedFigures(figure_a={...})
        new_category = category.to_dict()

        def _parse_classified_value(field: str, subfield_value: dict[str, list]) -> str:
            """
            :param field: 음고 a에서는 설정값으로 특정음정과 부분적 화음 구별을 할 수 없어 필드명이 필요
            :param subfield_value: ex) {'keyword': ['주요 리듬'], '주요 리듬': ['4분음표']}
            :return: ex) 4분음표
            """
            joined_value = "-".join([value[0] for value in subfield_value.values()])

            keyword, *preference = joined_value.split("-", maxsplit=1)

            if not preference:
                return keyword

            preference = preference[0]

            if keyword == "트랜지션":
                if preference == "unknown":
                    return keyword
                return f"{keyword}-{preference}"
            if keyword == "혼합된 형태":
                return keyword
            if keyword == "특수 주법":
                return keyword

            if field == "pitch_a":
                if keyword == "다성부" and preference != "unknown":
                    return preference
                return keyword
            
            return preference

        # classify
        for unclassified_field, unclassified_attr in category:
            print("unclassified_field:", unclassified_field, "unclassified_attr:", unclassified_attr)
            # ex) unclassified_field: figure_a, unclassified_attr: {'keyword': ['트랜지션']...}

            new_attr = new_category.get(unclassified_field)
            # ex) new_attr: {'keyword': ['트랜지션']...}

            for k, v in unclassified_attr.items():
                print('k=', k, 'v=', v)

                if k in PREFERENCE_AND_OR: # ex) k == '부분적 화음'
                    if k not in new_attr.get("keyword"):
                        # ex) 'keyword': ['단선율'] 이었다면,
                        new_attr.pop(k) # 뒤잇는 '부분적 화음' 키-밸류 페어 제거
                        continue

                    if PREFERENCE_AND_OR.get(k) in (AND, ANDOR):  # TODO: ANDOR case 분류 가능시 로직 분리
                        new_attr[k] = ["&".join(v)]
                        # ex) ['아르페지오', '빠른아르페지오', '글리산도'] -> ['아르페지오&빠른아르페지오&글리산도']
                        # 하지만 이 케이스는 아직 인코딩이 제대로 되지 않음

                    elif PREFERENCE_AND_OR.get(k) == SINGLE:
                        continue

                if v == ["unknown"]:
                    continue

                if v[0].endswith("optional"):
                    name = unclassified_field
                    classifier = getattr(Classifiers, name)
                    new_attr[k] = classifier.classify(sample, v)

                if len(new_attr[k]) > 1: 
                    # preference (세부 설정값, i.e., '트랜지션-OR' 처럼 독립된 column이 존재)가 있을 경우,
                    if k in PREFERENCE_AND_OR:
                        name = PREFERENCE_COLUMN_VARIABLE.get(k) # 네이밍을 영어로 변환 
                    else:
                        name = unclassified_field
                    track_category = sample.get("track_category")

                    # TODO: track_category에 따라 분류가 필요한 게 맞는지?
                    if name == "figure_a":
                        # figure_a 의 경우엔 'main_melody'냐, 'sub_melody'냐 에 따라서 타겟이 달라지기 때문에,
                        # ex) figure_a_main_melody의 타겟 종류: '주제적인', '솔로잉'
                        # ex) figure_a_sub_melody의 타겟 종류: '대선율', '트랜지션'
                        name = f"{name}_{track_category}" 

                    classifier = getattr(Classifiers, name)
                    new_attr[k] = classifier.classify(sample, v)
        # 정리
        for unclassified_field, unclassified_attr in new_category.items():
            new_category[unclassified_field] = _parse_classified_value(
                unclassified_field, unclassified_attr
            )

        # samples.append(dict(sample) | control | classified_guideline)

    print(len(samples))


import transformers
from torch.nn.modules.transformer import TransformerDecoderLayer
