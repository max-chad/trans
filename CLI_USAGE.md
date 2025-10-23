# Использование через командную строку

Проект можно запускать как через GUI, так и полностью из терминала. Ниже приведены рабочие команды и подсказки по настройке CLI.

## Запуск графического интерфейса

```powershell
python main.py
```

или явная форма:

```powershell
python main.py gui
```

В обоих случаях откроется окно приложения с экраном загрузки и основным интерфейсом.

## Базовая транскрибация

```powershell
python main.py transcribe <путь_к_видео_или_аудио> [опции]
```

Команда запускает воркер транскрибации и сохраняет результаты в выбранные форматы. Минимально требуется только путь к файлу.

### Аргументы `transcribe`

- `-o, --output-dir` — каталог, куда сохраняются итоговые файлы (по умолчанию рядом с исходником или из конфигурации).
- `-f, --format` — формат выгрузки; флаг можно повторять (доступны `srt`, `txt`, `txt_ts`, `vtt` и форматы, перечисленные в конфиге).
- `--timestamps` — добавляет временные метки в плоский текстовый вывод (`txt`).
- `-l, --language` — язык распознавания (`auto`, `en`, `ru` и т. д.).
- `-m, --model` — размер модели Whisper (`tiny`, `base`, `small`, `medium`, `large` и др.).
- `-d, --device` — устройство для инференса (`cpu`, `cuda`, `hybrid`, `auto`).
- `--no-correction` — отключает локальную постобработку текста (правки LLM).
- `--diarize` / `--no-diarize` — включение/отключение разметки итогового текста по спикерам.
- `--speaker-model` — идентификатор модели диаризации (например, `pyannote/speaker-diarization-3.1`).
- `--speaker-auth-token` — токен аутентификации Hugging Face для загрузки модели.
- `--speaker-min-speakers`, `--speaker-max-speakers` — минимальное/максимальное ожидаемое число спикеров (опционально).
- `--lmstudio-url`, `--lmstudio-model`, `--lmstudio-api-key` — параметры подключения к LM Studio, если коррекция делается внешней LLM.
- `--config` — путь к альтернативному конфигурационному JSON (по умолчанию `~/.video-transcriber/config.json`).
- `--quiet` — скрывает служебные сообщения, оставляя в выводе только пути к созданным файлам.

### Пример полного запуска

```powershell
python main.py transcribe data\interview.mp4 `
    -o exports `
    --format srt --format txt `
    --timestamps `
    --language en `
    --model small
```

Команда создаст каталог `exports`, выгрузит SRT и TXT версии расшифровки, добавит временные метки для текста и использует модель `small`.

## Настройка через конфиг

Базовые значения параметров берутся из `~/.video-transcriber/config.json`. В файле можно задать:

- предпочтительные форматы (`output_formats_multi` или `output_format`);
- параметры Whisper (`model_size`, `device`, `whisper_backend`, `faster_whisper_compute_type`);
- настройки параллелизма (`pipeline_mode`, `max_parallel_transcriptions`, `max_parallel_corrections`);
- параметры LM Studio (`lmstudio_*`) и локальной коррекции (`use_local_llm_correction`, `llama_*`).

CLI автоматически создаёт каталог назначения, если его нет, и использует настройки из конфига как значения по умолчанию.

## Программное использование

```python
from pathlib import Path
from app.cli import transcribe_file

results = transcribe_file(
    Path("data/interview.mp4"),
    output_dir=Path("exports"),
    formats=["srt", "txt"],
    include_timestamps=True,
    language="auto",
)

for path in results:
    print("Готовый файл:", path)
```

Дополнительно можно передать колбэки `progress_callback` и `log_callback`, чтобы получать уведомления о ходе работы, а также переопределить настройки LM Studio или путь к конфигу.

```python
def on_progress(value: int):
    print("Прогресс:", value, "%")

def on_log(level: str, message: str):
    print(level.upper(), message)

transcribe_file(
    Path("audio.wav"),
    progress_callback=on_progress,
    log_callback=on_log,
)
```

CLI использует тот же `TranscriptionWorker`, что и GUI, поэтому результаты идентичны независимо от выбранного интерфейса.
