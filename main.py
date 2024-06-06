import pathlib

from basic_pitch import ICASSP_2022_MODEL_PATH, inference
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_N_SEMITONES,
    FFT_HOP,
)

RESOURCES_PATH = pathlib.Path(__file__).parent / "tests/resources"


def run():
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    with RESOURCES_PATH as tmpdir:
        inference.predict_and_save(
            [test_audio_path],
            tmpdir,
            True,
            True,
            True,
            True,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )


if __name__ == "__main__":
    run()
