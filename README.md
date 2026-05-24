# text-image-aligment

**Нейросетевая система исправления искажений на изображениях с текстом**

- Направление: компьютерное зрение, обработка изображений.
- Основа работы: предыдущие курсовые по выравниванию текста.
- Тип работы: программно-исследовательская.
- Итог: автоматизированная система выравнивания текста с оценкой OCR-качества.

## Датасет и канон изображения

Разметка `deltaTPS` в `metadata.json` задаётся в координатах **файла warped на диске**. При обучении `TPSDataset` может принимать `spatial_spec` ([`CanvasSpatialSpec`](src/tps_dewarp/dataset/canvas_spatial.py)): тогда изображение и целевой `deltaTPS` **синхронно** приводятся к одному канвасу (`letterbox` или `stretch`), после чего применяется только фотометрия (`photometric_transform`, например `Normalize`). Инференс и `build_remap_from_delta_tps` нужно вызывать с теми же `H`, `W`, что у канона (см. `notebooks/dev/03_train_model.ipynb`, `05_eval_model.ipynb`). Старый режим с одним аргументом `transform` без `spatial_spec` сохранён для совместимости.

## Обучение (Trainer, YAML, MLflow)

- **Конфиг:** [`configs/train_default.yaml`](configs/train_default.yaml) — датасет, сплит, `DataLoader` (`drop_last`, `persistent_workers`, `prefetch_factor`), модель (`pretrained`, `tanh_output_scale`), веса лосса по `difficulty_weights`, оптимизатор, `warmup_epochs` + cosine, `use_amp` (только CUDA), `metric_for_best` (`val_loss` или `val_l2_px`), `grad_clip_norm`, tqdm, `l2_canvas_size`, чекпойнты, MLflow.
- **Smoke:** [`configs/train_smoke.yaml`](configs/train_smoke.yaml) — одна эпоха, малый batch для быстрой проверки пайплайна.
- **Кеш весов torchvision:** в ноутбуке задаётся `TORCH_HOME` в `./.cache/torch` внутри репозитория (см. `notebooks/dev/03_train_model.ipynb`).
- **Код:** пакет [`src/tps_dewarp/training/`](src/tps_dewarp/training/) — `load_train_config`, `build_tps_dataloaders`, `build_model`, класс `Trainer`.
- **Запуск из ноутбука:** `dagshub.init` и при необходимости `mlflow.set_experiment` / `mlflow.start_run` — до создания `Trainer`, затем `trainer.fit(resume_from=None)`. Для продолжения: `trainer.fit(resume_from="…/last.pt")` (или `best.pt`); в чекпойнте хранится `mlflow_run_id`, Trainer возобновляет тот же run.
- **Чекпойнты** (в `checkpoint.dir` из YAML): `last.pt` после каждой эпохи, `best.pt` при улучшении метрики из `training.metric_for_best`, `final.pt` в конце успешного `fit`. При `use_amp` на CUDA в чекпойнт дополнительно пишется `scaler_state_dict` для корректного resume.
- **Метрики эпохи:** помимо `train_loss` / `val_loss` / `train_l2_px` / `val_l2_px` — квантили `val_l2_px_p50|p95|max`, `val_loss_{difficulty}`, `val_l2_px_{difficulty}`, хвосты `val_l2_px_{difficulty}_p95|max`, `grad_norm_mean`, `epoch_time_sec`, `samples_per_sec`.
- **MLflow:** при улучшении best — опционально `log_artifact` для `best.pt`; в конце — один вызов `mlflow.pytorch.log_model` для финальной модели (без `registered_model_name`). **`mlflow.end_run`** вызывается только если этот же `fit` сам открыл run; если run уже активен (открыт в ноутбуке), Trainer его не закрывает. В параметры и теги run пишутся **`model`**, **`img_size`**, **`grid_size`** (короткие имена под таблицу DagsHub) плюс полный набор полей из YAML; при `resume` повторный `log_params` не вызывается (чтобы не дублировать ключи), но теги `model` / `img_size` / `grid_size` обновляются в начале `fit`. Метрики `train_loss` и `val_l2_px` появляются после первой завершённой эпохи.
