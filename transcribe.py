#!/usr/bin/env python
import os
import argparse
import tempfile
import math
import time
import concurrent.futures
from openai import OpenAI
from moviepy import VideoFileClip, AudioFileClip


def transcribe_audio(file_path):
    print(f"--- Запуск процесса транскрипции ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ОШИБКА] Переменная окружения OPENAI_API_KEY не установлена.")
        return

    print("[ИНФО] API-ключ найден. Инициализация клиента OpenAI...")
    client = OpenAI(api_key=api_key)

    if not os.path.exists(file_path):
        print(f"[ОШИБКА] Файл не найден по пути: {file_path}")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"[ИНФО] Файл: {os.path.basename(file_path)}")
    print(f"[ИНФО] Расширение: {file_ext}")
    print(f"[ИНФО] Размер: {file_size_mb:.2f} MB")

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
    MAX_SIZE_BYTES = 25 * 1024 * 1024
    temp_files = []

    try:
        is_video = file_ext in video_extensions

        # Если это видео или аудио файл больше лимита — обрабатываем через moviepy
        if is_video or file_size > MAX_SIZE_BYTES:
            if is_video:
                print(f"[ПРОЦЕСС] Обнаружен видеоформат. Извлечение аудиодорожки...")
            else:
                print(f"[ПРОЦЕСС] Файл превышает лимит 25MB. Оптимизация размера...")

            clip = None
            if is_video:
                clip = VideoFileClip(file_path)
                audio_clip = clip.audio
            else:
                audio_clip = AudioFileClip(file_path)

            if audio_clip is None:
                print("[ОШИБКА] В указанном файле не обнаружено аудиопотока.")
                if clip:
                    clip.close()
                return

            duration = audio_clip.duration
            print(f"[ИНФО] Длительность аудио: {duration:.2f} сек.")

            # Временный файл для извлеченного/сжатого аудио
            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            temp_files.append(temp_audio_path)

            print(f"[ПРОЦЕСС] Конвертация в MP3 (bitrate: 128k)...")
            audio_clip.write_audiofile(
                temp_audio_path, codec="libmp3lame", bitrate="128k", logger=None
            )

            processed_size = os.path.getsize(temp_audio_path)
            processed_size_mb = processed_size / (1024 * 1024)
            print(f"[ИНФО] Размер после сжатия: {processed_size_mb:.2f} MB")

            results = []

            if processed_size <= MAX_SIZE_BYTES:
                print("[ПРОЦЕСС] Отправка файла в OpenAI Whisper API...")
                start_time = time.time()
                with open(temp_audio_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model="whisper-1", file=f
                    )
                    results.append(response.text)
                print(
                    f"[УСПЕХ] Транскрипция завершена за {time.time() - start_time:.2f} сек."
                )
            else:
                # Если файл все еще больше 25MB, разбиваем на части
                num_chunks = math.ceil(processed_size / (MAX_SIZE_BYTES * 0.95))
                chunk_duration = duration / num_chunks
                print(f"[ВНИМАНИЕ] Файл все еще велик ({processed_size_mb:.2f} MB).")
                print(
                    f"[ПРОЦЕСС] Разбивка на {num_chunks} фрагментов по ~{chunk_duration:.1f} сек..."
                )

                chunk_paths = []
                for i in range(num_chunks):
                    start = i * chunk_duration
                    end = min((i + 1) * chunk_duration, duration)
                    print(
                        f"[ЧАСТЬ {i+1}/{num_chunks}] Подготовка фрагмента {start:.1f}с - {end:.1f}с..."
                    )

                    chunk_temp = tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    )
                    chunk_temp_path = chunk_temp.name
                    chunk_temp.close()
                    temp_files.append(chunk_temp_path)
                    chunk_paths.append(chunk_temp_path)

                    subclip = audio_clip.subclipped(start, end)
                    subclip.write_audiofile(
                        chunk_temp_path, codec="libmp3lame", bitrate="128k", logger=None
                    )

                print(
                    f"[ПРОЦЕСС] Отправка {len(chunk_paths)} фрагментов в API параллельно..."
                )
                start_parallel = time.time()

                def upload_to_whisper(path):
                    with open(path, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            model="whisper-1", file=f
                        )
                        return resp.text

                # Параллельное выполнение запросов
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # executor.map гарантирует сохранение порядка результатов
                    results = list(executor.map(upload_to_whisper, chunk_paths))

                print(
                    f"[УСПЕХ] Все части успешно обработаны за {time.time() - start_parallel:.2f} сек."
                )

            print("\n" + "=" * 30)
            print("ИТОГОВЫЙ ТЕКСТ:")
            print("-" * 30)
            print(" ".join(results))
            print("=" * 30 + "\n")

            audio_clip.close()
            if clip:
                clip.close()
        else:
            # Маленький аудиофайл отправляем сразу
            print(
                f"[ПРОЦЕСС] Файл небольшой, отправляем напрямую в OpenAI Whisper API..."
            )
            start_time = time.time()
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            print(
                f"[УСПЕХ] Транскрипция завершена за {time.time() - start_time:.2f} сек."
            )

            print("\n" + "=" * 30)
            print("ИТОГОВЫЙ ТЕКСТ:")
            print("-" * 30)
            print(transcription.text)
            print("=" * 30 + "\n")

    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА] Произошел сбой: {e}")
    finally:
        if temp_files:
            print(f"[ОЧИСТКА] Удаление временных файлов ({len(temp_files)} шт.)...")
            for path in temp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось удалить {path}: {e}")
        print("--- Работа завершена ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Транскрипция аудио и видео с помощью OpenAI Whisper API"
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        default="audio.mp3",
        help="Путь к аудио или видеофайлу (по умолчанию: audio.mp3)",
    )

    args = parser.parse_args()

    transcribe_audio(args.file_path)
