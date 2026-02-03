#!/usr/bin/env python
import os
import argparse
import tempfile
from openai import OpenAI
from moviepy import VideoFileClip


def transcribe_audio(file_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Ошибка: Переменная окружения OPENAI_API_KEY не установлена.")
        return

    client = OpenAI(api_key=api_key)

    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".flv",
        ".mpeg",
        ".mpg",
        ".wmv",
    }

    temp_audio_path = None

    try:
        # Если это видео — извлекаем аудио
        if file_ext in video_extensions:
            print(f"Обнаружен видеофайл. Извлекаю аудио...")
            video = VideoFileClip(file_path)

            if video.audio is None:
                print("Ошибка: В видеофайле не найдена аудиодорожка.")
                video.close()
                return

            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()

            video.audio.write_audiofile(
                temp_audio_path, codec="libmp3lame", verbose=False, logger=None
            )
            video.close()
            process_path = temp_audio_path
        else:
            process_path = file_path

        print(f"Начинаю транскрипцию файла: {file_path}...")

        with open(process_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

        print("\nРезультат транскрипции:")
        print("-" * 20)
        print(transcription.text)
        print("-" * 20)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        # Удаляем временный файл, если он был создан
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио и видео с помощью OpenAI Whisper API"
    )

    # Путь к файлу
    parser.add_argument(
        "file_path",
        nargs="?",
        default="audio.mp3",
        help="Путь к аудио или видеофайлу (по умолчанию: audio.mp3)",
    )

    args = parser.parse_args()

    transcribe_audio(args.file_path)
