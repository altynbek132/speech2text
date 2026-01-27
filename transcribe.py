import os
import openai
from openai import OpenAI
import argparse


def transcribe_audio(file_path, output_file):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return

    print(f"Начинаю транскрипцию файла: {file_path}...")

    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

        print("\nРезультат транскрипции:")
        print("-" * 20)
        print(transcription.text)
        print("-" * 20)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription.text)
        print(f"\nТекст успешно сохранен в {output_file}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио с помощью OpenAI Whisper API"
    )

    # Путь к аудиофайлу
    parser.add_argument(
        "file_path",
        nargs="?",
        default="audio.mp3",
        help="Путь к аудиофайлу (по умолчанию: audio.mp3)",
    )

    # Путь к выходному файлу
    parser.add_argument(
        "output_file",
        nargs="?",
        default="transcription.txt",
        help="Путь к файлу для сохранения (по умолчанию: transcription.txt)",
    )

    args = parser.parse_args()

    transcribe_audio(args.file_path, args.output_file)
