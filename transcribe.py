#!/usr/bin/env python
import os
import argparse
import tempfile
import math
from openai import OpenAI
from moviepy import VideoFileClip, AudioFileClip


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

    # Лимит OpenAI Whisper API — 25 MB
    MAX_SIZE_BYTES = 20 * 1024 * 1024
    temp_files = []

    try:
        file_size = os.path.getsize(file_path)
        is_video = file_ext in video_extensions

        # Если это видео или аудио файл больше лимита — обрабатываем через moviepy
        if is_video or file_size > MAX_SIZE_BYTES:
            print(f"Обработка файла (размер: {file_size / 1024 / 1024:.2f} MB)...")

            clip = None
            if is_video:
                clip = VideoFileClip(file_path)
                audio_clip = clip.audio
            else:
                audio_clip = AudioFileClip(file_path)

            if audio_clip is None:
                print("Ошибка: В файле не найдена аудиодорожка.")
                if clip:
                    clip.close()
                return

            duration = audio_clip.duration

            # Временный файл для извлеченного/сжатого аудио
            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            temp_files.append(temp_audio_path)

            print("Конвертация в MP3...")
            # Используем 128k битрейт для уменьшения размера без сильной потери качества
            audio_clip.write_audiofile(
                temp_audio_path, codec="libmp3lame", bitrate="128k", logger=None
            )

            processed_size = os.path.getsize(temp_audio_path)
            results = []

            if processed_size <= MAX_SIZE_BYTES:
                print("Начинаю транскрипцию...")
                with open(temp_audio_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model="whisper-1", file=f
                    )
                    results.append(response.text)
            else:
                # Если файл все еще больше 25MB, разбиваем на части
                num_chunks = math.ceil(processed_size / (MAX_SIZE_BYTES * 0.95))
                chunk_duration = duration / num_chunks
                print(f"Файл все еще велик ({processed_size / 1024 / 1024:.2f} MB).")
                print(f"Разбиваю на {num_chunks} частей...")

                for i in range(num_chunks):
                    start = i * chunk_duration
                    end = min((i + 1) * chunk_duration, duration)
                    print(
                        f"Обработка части {i+1}/{num_chunks} ({start:.0f}с - {end:.0f}с)..."
                    )

                    chunk_temp = tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    )
                    chunk_temp_path = chunk_temp.name
                    chunk_temp.close()
                    temp_files.append(chunk_temp_path)

                    subclip = audio_clip.subclipped(start, end)
                    subclip.write_audiofile(
                        chunk_temp_path, codec="libmp3lame", bitrate="128k", logger=None
                    )

                    with open(chunk_temp_path, "rb") as f:
                        response = client.audio.transcriptions.create(
                            model="whisper-1", file=f
                        )
                        results.append(response.text)

                    subclip.close()

            print("\nРезультат транскрипции:")
            print("-" * 20)
            print(" ".join(results))
            print("-" * 20)

            audio_clip.close()
            if clip:
                clip.close()
        else:
            # Маленький аудиофайл отправляем сразу
            print(f"Начинаю транскрипцию файла: {file_path}...")
            with open(file_path, "rb") as audio_file:
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
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
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
