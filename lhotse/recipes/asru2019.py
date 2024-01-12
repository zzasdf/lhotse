"""
About the Bengali.AI Speech corpus

The competition dataset comprises about 1200 hours of recordings of Bengali speech.
Your goal is to transcribe recordings of speech that is out-of-distribution with respect to the training set.

Note that this is a Code Competition, in which the actual test set is hidden.
In this public version, we give some sample data in the correct format to help you author your solutions.
The full test set contains about 20 hours of speech in almost 8000 MP3 audio files.
All of the files in the test set are encoded at a sample rate of 32k, a bit rate of 48k, in one channel.

It is covered in more detail at https://arxiv.org/abs/2305.09688

Please download manually by
kaggle competitions download -c bengaliai-speech
"""

import logging
import os
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    get_ffmpeg_torchaudio_info_enabled,
    set_ffmpeg_torchaudio_info_enabled,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
import re

ASRU2019 = ("train200", "train500", "dev", "dev2", "test")


@contextmanager
def disable_ffmpeg_torchaudio_info() -> None:
    enabled = get_ffmpeg_torchaudio_info_enabled()
    set_ffmpeg_torchaudio_info_enabled(False)
    try:
        yield
    finally:
        set_ffmpeg_torchaudio_info_enabled(enabled)


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    audio_id: str,
    text: Optional[str] = "",
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=audio_id,
    )
    segment = SupervisionSegment(
        id=audio_id,
        recording_id=audio_id,
        text=text,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Chinese",
    )

    return recording, segment

def normalization(text:str):
    pattern = re.compile("\[.\]")
    text = pattern.sub("", text).replace("~", "").strip()
    return text

def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    if subset == "train200":
        part_path = corpus_dir / "200小时中英混读手机采集语音数据"
    elif subset == "train500":
        part_path = corpus_dir / "500小时普通话手机采集语音数据"
    elif subset == "dev":
        part_path = corpus_dir / "ASRU100-fixed-dev"
    elif subset == "dev2":
        part_path = corpus_dir / "ASRUn20"
    else:
        part_path = corpus_dir / "release"
    
    if subset == "test":
        with open(part_path/"ASRU_test-text.txt", "r") as f:
            lines = f.readlines()
            lines = [item.split() for item in lines]
            audio_info = {item[0].replace(".wav", ""): ' '.join(item[1:]).strip() for item in lines}
    else:
        audio_info = dict()
        text_paths = list(part_path.rglob("*.txt"))
        for item in text_paths:
            audio_id = os.path.split(str(item))[1].replace(".txt", "")
            with open(item, 'r') as f:
                audio_info[audio_id] = f.readlines()[0].strip()

    audio_paths = list(part_path.rglob("*.wav"))

    with disable_ffmpeg_torchaudio_info():
        with ProcessPoolExecutor(num_jobs) as ex:
            futures = []
            recordings = []
            supervisions = []
            for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
                audio_id = os.path.split(str(audio_path))[1].replace(".wav", "")
                if audio_info is not None:
                    if audio_id not in audio_info.keys():
                        continue
                    text = normalization(audio_info[audio_id])
                else:
                    text = None
                futures.append(
                    ex.submit(_parse_utterance, corpus_dir, audio_path, audio_id, text)
                )

            for future in tqdm(futures, desc="Processing"):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

    return recording_set, supervision_set


def prepare_asru2019(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the Bengali.AI Speech dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing ASRU2019 Codeswitching Speech...")

    subsets = ASRU2019

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing ASRU2019 Codeswitching subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="asru2019",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"ASRU2019 Codeswitching subset: {part} already prepared - skipping."
            )
            continue

        if part == "train":
            recording_set, supervision_set = _prepare_subset(
                part, corpus_dir, num_jobs
            )
        elif part == "valid":
            recording_set, supervision_set = _prepare_subset(
                part, corpus_dir, num_jobs
            )
        elif part == "valid2":
            recording_set, supervision_set = _prepare_subset(
                part, corpus_dir, num_jobs
            )
        else:
            recording_set, supervision_set = _prepare_subset(
                part, corpus_dir, num_jobs
            )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"asru2019_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"asru2019_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_asru2019("/star-data/kangwei/data/bengaliai", ".", 8)