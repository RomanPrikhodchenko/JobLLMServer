# vLLM + LiteLLM API Gateway

Высокопроизводительный сервер LLM с API Gateway для управления ключами, лимитами и бюджетами.

## Архитектура

```
Client -> LiteLLM (4000) -> vLLM (8000) -> GPU
              |
          PostgreSQL (5432)
```

- **vLLM** - быстрый inference сервер с OpenAI-совместимым API
- **LiteLLM** - API Gateway с rate limiting и управлением ключами
- **PostgreSQL** - хранение ключей, статистики, бюджетов

## Быстрый старт

### Требования

- Docker Desktop с поддержкой GPU
- NVIDIA GPU + CUDA драйверы
- NVIDIA Container Toolkit

### 1. Настройка .env

Скопируйте `.env.example` в `.env` и настройте:

```bash
# Обязательные ключи безопасности (сгенерируйте свои!)
LITELLM_MASTER_KEY=sk-ваш-мастер-ключ-минимум-32-символа
LITELLM_SALT_KEY=ваш-salt-ключ-base64
POSTGRES_PASSWORD=надежный-пароль-базы-данных
UI_PASSWORD=пароль-для-веб-интерфейса

# Модель (см. раздел "Выбор модели")
VLLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
SERVED_MODEL_NAME=qwen2.5-0.5b

# GPU настройки (см. раздел "Настройка GPU")
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
```

### 2. Запуск

```bash
docker compose up -d
```

### 3. Проверка

```bash
# Статус контейнеров
docker ps

# Логи
docker compose logs -f

# Тест API
curl http://localhost:8000/v1/models
```

## Доступ

| Сервис | URL | Описание |
|--------|-----|----------|
| LiteLLM API | http://localhost:4000 | Основной API endpoint |
| LiteLLM UI | http://localhost:4000/ui | Веб-интерфейс (admin / UI_PASSWORD) |
| vLLM API | http://localhost:8000 | Прямой доступ к vLLM |
| Swagger | http://localhost:4000/docs | API документация |

---

## Настройка .env

### Ключи безопасности

```bash
# Master ключ для управления LiteLLM (ОБЯЗАТЕЛЬНО начинается с sk-)
LITELLM_MASTER_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Salt для шифрования API ключей (НИКОГДА не меняйте после создания ключей!)
LITELLM_SALT_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Пароль PostgreSQL
POSTGRES_PASSWORD=xxxxxxxxxxxxx

# Пароль для веб-интерфейса
UI_PASSWORD=xxxxxxxxxxxxx
```

### Генерация ключей (PowerShell)

```powershell
# Master Key
"sk-" + -join ((1..40) | ForEach-Object { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[(Get-Random -Maximum 62)] })

# Salt Key
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))

# Password
-join ((1..24) | ForEach-Object { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[(Get-Random -Maximum 62)] })
```

### Настройки модели

```bash
# HuggingFace токен (нужен только для gated моделей, например Llama)
HF_TOKEN=hf_xxxxxxxxxxxxx

# Модель из HuggingFace
VLLM_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

# Имя модели в API (используется в запросах)
SERVED_MODEL_NAME=qwen2.5-0.5b

# Максимальная длина контекста
MAX_MODEL_LEN=4096
```

---

## Выбор модели

### Открытые модели (без HF_TOKEN)

| Модель | Размер | VRAM | Описание |
|--------|--------|------|----------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | 2-4 GB | Для тестов |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | 4-6 GB | Легкая |
| `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | 16-20 GB | Средняя |
| `Qwen/Qwen3-8B` | ~16 GB | 18-24 GB | Хорошее качество |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~15 GB | 16-20 GB | Популярная |

### Gated модели (нужен HF_TOKEN)

| Модель | Размер | VRAM | Описание |
|--------|--------|------|----------|
| `meta-llama/Llama-3.2-1B-Instruct` | ~2 GB | 4-6 GB | Легкая Llama |
| `meta-llama/Llama-3.2-3B-Instruct` | ~6 GB | 8-12 GB | Средняя Llama |
| `meta-llama/Meta-Llama-3-8B-Instruct` | ~16 GB | 18-24 GB | Полная Llama 3 |

Для gated моделей:
1. Зарегистрируйтесь на https://huggingface.co
2. Примите лицензию модели на её странице
3. Создайте токен: https://huggingface.co/settings/tokens
4. Укажите в `.env`: `HF_TOKEN=hf_xxxxx`

### Добавление новой модели

1. Измените `.env`:
```bash
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
SERVED_MODEL_NAME=qwen2.5-7b
MAX_MODEL_LEN=8192
```

2. Обновите `config/litellm/config.yaml`:
```yaml
model_list:
  - model_name: qwen2.5-7b
    litellm_params:
      model: openai/qwen2.5-7b
      api_base: http://vllm:8000/v1
      api_key: dummy
```

3. Перезапустите:
```bash
docker compose down
docker compose up -d
```

---

## Несколько моделей одновременно

**Важно:** vLLM загружает ОДНУ модель при старте и держит её в VRAM. Динамическая загрузка моделей "на лету" не поддерживается.

LiteLLM — это только роутер/прокси, он не управляет загрузкой моделей в vLLM.

```
LiteLLM (роутер) ──→ vLLM Instance 1 (модель A)
                ──→ vLLM Instance 2 (модель B)
                ──→ OpenAI API (внешний)
```

### Вариант 1: Несколько vLLM инстансов

Добавьте в `docker-compose.yml` дополнительные vLLM сервисы:

```yaml
  vllm-small:
    image: vllm/vllm-openai:latest
    container_name: vllm-small
    runtime: nvidia
    ipc: host
    environment:
      HF_HOME: /models
    command: >
      --model Qwen/Qwen2.5-0.5B-Instruct
      --served-model-name qwen-small
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.3
      --max-model-len 4096
    volumes:
      - F:/vllm-models:/models
    ports:
      - "8001:8000"
    networks:
      - vllm-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  vllm-large:
    image: vllm/vllm-openai:latest
    container_name: vllm-large
    runtime: nvidia
    ipc: host
    environment:
      HF_HOME: /models
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --served-model-name qwen-large
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.6
      --max-model-len 8192
    volumes:
      - F:/vllm-models:/models
    ports:
      - "8002:8000"
    networks:
      - vllm-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Обновите `config/litellm/config.yaml`:

```yaml
model_list:
  - model_name: qwen-small
    litellm_params:
      model: openai/qwen-small
      api_base: http://vllm-small:8000/v1
      api_key: dummy

  - model_name: qwen-large
    litellm_params:
      model: openai/qwen-large
      api_base: http://vllm-large:8000/v1
      api_key: dummy
```

Теперь в API можно выбирать модель:
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-ключ" \
  -d '{"model": "qwen-small", "messages": [...]}'

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-ключ" \
  -d '{"model": "qwen-large", "messages": [...]}'
```

**Важно:** Обе модели занимают VRAM постоянно. Распределите `gpu-memory-utilization` так, чтобы суммарно не превышало ~0.95.

### Вариант 2: Смена модели перезапуском

Если нужна только одна модель в момент времени:

```bash
# Изменить .env
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
SERVED_MODEL_NAME=qwen-large

# Перезапустить vLLM
docker compose restart vllm
```

Модель скачается автоматически при первом запуске (если ещё не скачана).

### Вариант 3: LiteLLM + внешние API

LiteLLM может роутить на внешние сервисы параллельно с локальным vLLM:

```yaml
model_list:
  # Локальная модель
  - model_name: local-qwen
    litellm_params:
      model: openai/qwen2.5-0.5b
      api_base: http://vllm:8000/v1
      api_key: dummy

  # OpenAI
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: sk-openai-ключ

  # Anthropic
  - model_name: claude-3
    litellm_params:
      model: claude-3-sonnet-20240229
      api_key: sk-ant-ключ
```

---

## Настройка GPU

### Основные параметры

```bash
# Количество GPU для tensor parallelism
# 1 = одна GPU, 2 = две GPU работают вместе
TENSOR_PARALLEL_SIZE=1

# Сколько VRAM использовать (0.0 - 1.0)
# 0.9 = 90% памяти GPU
# Уменьшите если ошибка Out of Memory
GPU_MEMORY_UTILIZATION=0.9

# Максимальная длина контекста (в токенах)
# Больше контекст = больше VRAM
MAX_MODEL_LEN=4096
```

### Рекомендации по VRAM

| VRAM | Рекомендуемые модели | MAX_MODEL_LEN |
|------|---------------------|---------------|
| 6-8 GB | 0.5B - 1.5B | 2048 - 4096 |
| 12 GB | до 3B | 4096 - 8192 |
| 16 GB | до 7B | 4096 - 8192 |
| 24 GB | до 8B | 8192 - 16384 |
| 48 GB | до 14B | 16384 - 32768 |
| 80 GB | до 70B | 32768+ |

### Multi-GPU (несколько видеокарт)

Для моделей которые не помещаются на одну GPU:

```bash
# 2 GPU
TENSOR_PARALLEL_SIZE=2

# 4 GPU
TENSOR_PARALLEL_SIZE=4
```

Требования:
- Все GPU должны быть одинаковые (или очень похожие)
- NVLink улучшает производительность (но не обязателен)

### Оптимизация производительности

В `docker-compose.yml` можно добавить флаги vLLM:

```yaml
command: >
  --model ${VLLM_MODEL_NAME}
  --served-model-name ${SERVED_MODEL_NAME}
  --host 0.0.0.0
  --port 8000
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1}
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.9}
  --max-model-len ${MAX_MODEL_LEN:-4096}
  --trust-remote-code
  --enable-chunked-prefill          # Быстрее для длинных промптов
  --max-num-batched-tokens 8192     # Больше батч = выше throughput
  --max-num-seqs 256                # Макс параллельных запросов
```

### Решение проблем с GPU

**Out of Memory:**
```bash
# Уменьшите использование памяти
GPU_MEMORY_UTILIZATION=0.8

# Или уменьшите контекст
MAX_MODEL_LEN=2048
```

**Проверка GPU:**
```bash
# Статус GPU
nvidia-smi

# Тест Docker + GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

---

## Хранение моделей

По умолчанию модели скачиваются на диск F: (настроено в `docker-compose.yml`):

```yaml
volumes:
  - F:/vllm-models:/models
```

Чтобы изменить путь, отредактируйте эту строку.

---

## Использование API

### Создание API ключа

Через веб-интерфейс: http://localhost:4000/ui

Или через API:
```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer ВАШ_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "my-app",
    "max_budget": 100,
    "duration": "30d"
  }'
```

### Запрос к модели

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-ваш-виртуальный-ключ",
    base_url="http://localhost:4000/v1"
)

response = client.chat.completions.create(
    model="qwen2.5-0.5b",  # SERVED_MODEL_NAME из .env
    messages=[{"role": "user", "content": "Привет!"}]
)

print(response.choices[0].message.content)
```

### curl пример

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-ваш-ключ" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Привет!"}]
  }'
```

---

## Управление

```bash
# Запуск
docker compose up -d

# Остановка
docker compose down

# Перезапуск
docker compose restart

# Логи
docker compose logs -f
docker compose logs -f vllm
docker compose logs -f litellm

# Сброс базы данных (удалит все ключи!)
docker compose down
rm -rf ./data/postgres
docker compose up -d
```

---

## Структура проекта

```
vLLM/
├── docker-compose.yml      # Конфигурация контейнеров
├── .env                    # Настройки (не в git!)
├── .env.example            # Шаблон настроек
├── config/
│   └── litellm/
│       └── config.yaml     # Конфигурация LiteLLM
└── data/                   # Данные (не в git!)
    ├── postgres/           # База данных
    └── logs/               # Логи
```
