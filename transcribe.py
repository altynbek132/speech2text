import os
import openai
from openai import OpenAI

def transcribe_audio(file_path):
    # Инициализация клиента OpenAI
    # Убедитесь, что переменная окружения OPENAI_API_KEY установлена
    # Или замените 'os.getenv("OPENAI_API_KEY")' на ваш ключ строкой
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return

    print(f"Начинаю транскрипцию файла: {file_path}...")

    try:
        with open(file_path, "rb") as audio_file:
            # Вызов Whisper API
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        # Вывод результата
        print("\nРезультат транскрипции:")
        print("-" * 20)
        print(transcription.text)
        print("-" * 20)
        
        # Сохранение в текстовый файл
        output_file = "transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription.text)
        print(f"\nТекст успешно сохранен в {output_file}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # В вашем дереве файлов указан audio.mp3
    FILE_TO_TRANSCRIBE = "audio.mp3"
    transcribe_audio(FILE_TO_TRANSCRIBE)
