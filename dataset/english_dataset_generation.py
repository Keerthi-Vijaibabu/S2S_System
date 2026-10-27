# pip install torch torchaudio soundfile numpy pandas
# pip install datasets torchcodec
# pip install speechbrain sentencepiece sacremoses
# pip install transformers accelerate



"""
Tamil -> English Speech-to-Speech Translation
=============================================

Dataset: AI4Bharat IndicVoices (Tamil)

Pipeline:
1. Load Tamil audio and its transcript.
2. Extract a speaker x-vector from Tamil audio.
3. Translate Tamil text to English using NLLB-200.
4. Synthesize English audio using SpeechT5.
5. Save audio, embeddings, transcripts and CSV manifest.

NOTE:
- This is a cascaded S2ST baseline.
- Speaker identity preservation is experimental.
- The source transcript is supplied by IndicVoices.
"""

import csv
import json
import traceback
from pathlib import Path
import io

import numpy as np
import soundfile as sf
import torch
import torchaudio

from datasets import load_dataset, Audio

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)



# ============================================================
# STEP 0: CONFIGURATION
# ============================================================

# Number of successfully processed samples.
NUM_SAMPLES = 10

# Dataset split.
# Use "train" for training data generation later.
DATASET_SPLIT = "valid"

# Use GPU if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models.
TRANSLATION_MODEL = (
    "facebook/nllb-200-distilled-600M"
)

SPEAKER_MODEL = (
    "speechbrain/spkrec-xvect-voxceleb"
)

TTS_MODEL = "microsoft/speecht5_tts"
VOCODER_MODEL = "microsoft/speecht5_hifigan"

# Output directories.
OUTPUT_DIR = Path("tamil_english_s2st")

TAMIL_AUDIO_DIR = OUTPUT_DIR / "tamil_audio"
ENGLISH_AUDIO_DIR = OUTPUT_DIR / "english_audio"
EMBEDDING_DIR = OUTPUT_DIR / "speaker_embeddings"
TEXT_DIR = OUTPUT_DIR / "transcripts"

for directory in [
    OUTPUT_DIR,
    TAMIL_AUDIO_DIR,
    ENGLISH_AUDIO_DIR,
    EMBEDDING_DIR,
    TEXT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

print("Device:", DEVICE)


# ============================================================
# STEP 1: LOAD THE TAMIL DATASET
# ============================================================
def load_tamil_dataset():
    print("\nLoading IndicVoices Tamil dataset...")

    dataset = load_dataset(
        "ai4bharat/IndicVoices",
        "tamil",
        split=DATASET_SPLIT,
        streaming=True,
    )

    # Prevent Hugging Face Datasets from invoking TorchCodec.
    # The audio column will provide raw bytes or a path instead.
    dataset = dataset.cast_column(
        "audio_filepath",
        Audio(decode=False)
    )

    return dataset


# ============================================================
# STEP 2: INITIALIZE ALL PRETRAINED MODELS
# ============================================================

def load_models():
    """
    Load the three main components:

    A. Speaker encoder:
       Extracts a 512-dimensional speaker x-vector.

    B. NLLB-200:
       Translates Tamil text into English.

    C. SpeechT5 + HiFi-GAN:
       Generates English speech conditioned on the x-vector.
    """

    print("\nLoading speaker encoder...")

    speaker_encoder = EncoderClassifier.from_hparams(
        source=SPEAKER_MODEL,
        savedir="pretrained_models/speaker_encoder",
        run_opts={"device": DEVICE},
        local_strategy=LocalStrategy.COPY,
    )

    print("Loading NLLB-200...")

    translation_tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL,
    )

    translation_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL,
        )
        .to(DEVICE)
        .eval()
    )


    print("Loading SpeechT5...")

    tts_processor = SpeechT5Processor.from_pretrained(
        TTS_MODEL
    )

    tts_model = (
        SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL
        )
        .to(DEVICE)
        .eval()
    )

    print("Loading HiFi-GAN vocoder...")

    vocoder = (
        SpeechT5HifiGan.from_pretrained(
            VOCODER_MODEL
        )
        .to(DEVICE)
        .eval()
    )

    return {
        "speaker_encoder": speaker_encoder,
        "translation_tokenizer": translation_tokenizer,
        "translation_model": translation_model,
        "tts_processor": tts_processor,
        "tts_model": tts_model,
        "vocoder": vocoder,
    }


# ============================================================
# STEP 3: EXTRACT AUDIO FROM DATASET
# ============================================================

def extract_audio(sample):
    """
    Decode IndicVoices audio using soundfile rather than TorchCodec.

    Supports embedded audio bytes and local file paths.
    Returns mono float32 audio and its sample rate.
    """

    audio = sample["audio_filepath"]

    if not isinstance(audio, dict):
        raise TypeError(
            f"Unexpected audio format: {type(audio)}"
        )

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes is not None:
        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )

    elif audio_path is not None:
        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )

    else:
        raise ValueError(
            "Audio contains neither bytes nor a valid path."
        )

    # soundfile returns [samples, channels].
    # Average the channels to produce mono audio.
    waveform = waveform.mean(axis=1)

    if waveform.size == 0:
        raise ValueError("Empty audio sample")

    if not np.all(np.isfinite(waveform)):
        raise ValueError("Invalid audio values")

    return waveform.astype(np.float32), sample_rate

# ============================================================
# STEP 4: PREPROCESS AUDIO FOR SPEAKER ENCODER
# ============================================================

def preprocess_audio(waveform, sample_rate):
    """
    The speaker encoder expects 16 kHz audio.

    Operations:
      1. Convert NumPy array to Torch tensor.
      2. Resample to 16 kHz.
      3. Return tensor shaped [1, samples].
    """

    waveform = torch.tensor(
        waveform, dtype=torch.float32
    ).unsqueeze(0)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=16000,
        )

    return waveform


# ============================================================
# STEP 5: EXTRACT SPEAKER EMBEDDING
# ============================================================

def extract_speaker_embedding(
    waveform,
    speaker_encoder,
):
    """
    Extract speaker characteristics from Tamil speech.

    Input:
        Tamil waveform at 16 kHz.

    Output:
        A 512-dimensional x-vector with shape [1, 512].

    This is a learned speaker representation, not a
    direct measurement of pitch or gender.

    SpeechT5 uses the embedding to condition the voice.
    """

    waveform = waveform.to(DEVICE)

    with torch.inference_mode():
        embedding = speaker_encoder.encode_batch(
            waveform
        )

    # Convert [1, 1, 512] to [1, 512].
    embedding = embedding.reshape(1, -1)

    if embedding.shape != (1, 512):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    # Normalize to unit length.
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=-1,
    )

    return embedding


# ============================================================
# STEP 6: TRANSLATE TAMIL TEXT INTO ENGLISH
# ============================================================

def translate_tamil_to_english(text, models):
    """
    Translate Tamil transcription to English.

    Tamil language code: tam_Taml
    English language code: eng_Latn

    Uses Meta NLLB-200 distilled 600M multilingual translation model.
    """

    tokenizer = models["translation_tokenizer"]
    translator = models["translation_model"]
    # NLLB tokenizer uses language tags to identify the source language.
    tokenizer.src_lang = "tam_Taml"

    # Tokenize Tamil text directly; no IndicTransToolkit preprocessing.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # Force the first generated token to select English output.
    english_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    if english_token_id == tokenizer.unk_token_id:
        raise RuntimeError("English language token missing from NLLB tokenizer")

    with torch.inference_mode():
        output = translator.generate(
            **inputs,
            forced_bos_token_id=english_token_id,
            num_beams=4,
            max_new_tokens=256,
        )

    # Convert generated token IDs to human-readable English.
    return tokenizer.batch_decode(
        output, skip_special_tokens=True
    )[0].strip()



# ============================================================
# STEP 7: SYNTHESIZE ENGLISH SPEECH
# ============================================================

def synthesize_english(
    english_text,
    speaker_embedding,
    models,
):
    """
    Convert translated English text into speech.

    Inputs:
      1. English text
      2. Speaker embedding from original Tamil audio

    SpeechT5 generates an acoustic representation.
    HiFi-GAN converts it into an audio waveform.

    Output sample rate: 16000 Hz.

    Note:
      Speaker identity preservation across languages
      is not guaranteed.
    """

    processor = models["tts_processor"]
    tts_model = models["tts_model"]
    vocoder = models["vocoder"]

    inputs = processor(
        text=english_text,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(DEVICE)

    with torch.inference_mode():
        speech = tts_model.generate_speech(
            input_ids,
            speaker_embeddings=speaker_embedding.to(DEVICE),
            vocoder=vocoder,
        )

    return speech.cpu().numpy()


# ============================================================
# STEP 8: SAVE OUTPUTS
# ============================================================

def save_outputs(
    sample_id,
    speaker_id,
    tamil_text,
    english_text,
    tamil_audio,
    tamil_sample_rate,
    english_audio,
    speaker_embedding,
):
    """
    Save all artifacts needed for S2ST experiments.

    Files saved per sample:
      - Tamil WAV
      - English WAV
      - Speaker embedding (.pt)
      - Tamil transcription (.txt)
      - English translation (.txt)
      - Metadata (.json)
    """

    name = f"sample_{sample_id:05d}"

    tamil_path = TAMIL_AUDIO_DIR / f"{name}.wav"
    english_path = ENGLISH_AUDIO_DIR / f"{name}.wav"
    embedding_path = EMBEDDING_DIR / f"{name}.pt"

    tamil_text_path = TEXT_DIR / f"{name}_ta.txt"
    english_text_path = TEXT_DIR / f"{name}_en.txt"

    metadata_path = TEXT_DIR / f"{name}.json"

    # Save original Tamil audio.
    sf.write(
        tamil_path,
        tamil_audio,
        tamil_sample_rate,
    )

    # Save synthesized English audio.
    sf.write(
        english_path,
        english_audio,
        16000,
    )

    # Save extracted speaker embedding.
    torch.save(
        speaker_embedding.cpu(),
        embedding_path,
    )

    # Save both transcripts as UTF-8.
    tamil_text_path.write_text(
        tamil_text, encoding="utf-8"
    )

    english_text_path.write_text(
        english_text, encoding="utf-8"
    )

    metadata = {
        "sample_id": sample_id,
        "speaker_id": speaker_id,
        "source_language": "tam_Taml",
        "target_language": "eng_Latn",
        "tamil_text": tamil_text,
        "english_text": english_text,
        "tamil_audio_path": str(tamil_path),
        "english_audio_path": str(english_path),
        "speaker_embedding_path": str(embedding_path),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


# ============================================================
# STEP 9: PROCESS THE DATASET
# ============================================================

def main():
    """
    Execute the complete S2ST pipeline.

    For each Tamil sample:
      1. Read Tamil transcription.
      2. Decode Tamil audio.
      3. Extract speaker embedding.
      4. Translate Tamil to English.
      5. Synthesize English speech.
      6. Save all outputs.
      7. Append a record to CSV.

    Errors are logged and the next sample is attempted.
    """

    dataset = load_tamil_dataset()
    models = load_models()

    # Create CSV manifest with a header.
    fieldnames = [
        "sample_id",
        "speaker_id",
        "tamil_text",
        "english_text",
        "tamil_audio_path",
        "english_audio_path",
        "speaker_embedding_path",
    ]

    # Resume support: skip already completed sample indices.
    completed = set()

    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(
            "r", encoding="utf-8", newline=""
        ) as f:
            for row in csv.DictReader(f):
                completed.add(int(row["sample_id"]))

    with MANIFEST_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        if MANIFEST_PATH.stat().st_size == 0:
            writer.writeheader()

        successful = len(completed)

        if successful >= NUM_SAMPLES:
            print("Requested samples already processed.")
            return

        # Enumerate keeps stable sample IDs across restarts.
        for index, sample in enumerate(dataset):

            if index in completed:
                continue

            print("\n" + "=" * 60)
            print("Processing dataset index:", index)

            try:
                # Extract provided Tamil transcription.
                tamil_text = (
                    sample.get("normalized")
                    or sample.get("text")
                    or sample.get("verbatim")
                    or ""
                ).strip()

                if not tamil_text:
                    print("Skipping empty transcript")
                    continue

                speaker_id = sample.get(
                    "speaker_id", "unknown"
                )

                print("Tamil:", tamil_text)
                print("Speaker:", speaker_id)

                # Decode the Tamil audio.
                audio, sample_rate = extract_audio(
                    sample
                )

                # Prepare 16 kHz audio for speaker encoder.
                processed_audio = preprocess_audio(
                    audio,
                    sample_rate,
                )

                # Extract the original speaker x-vector.
                embedding = extract_speaker_embedding(
                    processed_audio,
                    models["speaker_encoder"],
                )

                print(
                    "Embedding shape:",
                    tuple(embedding.shape),
                )

                # Translate Tamil transcript to English.
                english_text = translate_tamil_to_english(
                    tamil_text,
                    models,
                )

                if not english_text:
                    raise ValueError("Empty translation")

                print("English:", english_text)

                # Generate English speech using Tamil speaker.
                english_audio = synthesize_english(
                    english_text,
                    embedding,
                    models,
                )

                # Save outputs and metadata.
                metadata = save_outputs(
                    sample_id=index,
                    speaker_id=speaker_id,
                    tamil_text=tamil_text,
                    english_text=english_text,
                    tamil_audio=audio,
                    tamil_sample_rate=sample_rate,
                    english_audio=english_audio,
                    speaker_embedding=embedding,
                )

                # Update CSV manifest after successful save.
                writer.writerow({
                    key: metadata[key]
                    for key in fieldnames
                })

                csvfile.flush()

                completed.add(index)
                successful += 1

                print("Saved sample:", index)
                print(
                    f"Progress: {successful}/{NUM_SAMPLES}"
                )

                if successful >= NUM_SAMPLES:
                    break

            except Exception as error:
                print("Failed sample:", index)
                print("Error:", repr(error))
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("PROCESSING FINISHED")
    print("Successful samples:", successful)
    print("Manifest:", MANIFEST_PATH)
    print("Output directory:", OUTPUT_DIR)


# ============================================================
# STEP 10: ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
