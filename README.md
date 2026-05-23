# text-image-aligment

**Нейросетевая система исправления искажений на изображениях с текстом**

- Направление: компьютерное зрение, обработка изображений.
- Основа работы: предыдущие курсовые по выравниванию текста.
- Тип работы: программно-исследовательская.
- Итог: автоматизированная система выравнивания текста с оценкой OCR-качества.

## Датасет и канон изображения

Разметка `deltaTPS` в `metadata.json` задаётся в координатах **файла warped на диске**. При обучении `TPSDataset` может принимать `spatial_spec` ([`CanvasSpatialSpec`](src/tps_dewarp/dataset/canvas_spatial.py)): тогда изображение и целевой `deltaTPS` **синхронно** приводятся к одному канвасу (`letterbox` или `stretch`), после чего применяется только фотометрия (`photometric_transform`, например `Normalize`). Инференс и `build_remap_from_delta_tps` нужно вызывать с теми же `H`, `W`, что у канона (см. `notebooks/dev/03_train_model.ipynb`, `05_eval_model.ipynb`). Старый режим с одним аргументом `transform` без `spatial_spec` сохранён для совместимости.

## Обучение (Trainer, YAML, MLflow)

- **Конфиг:** [`configs/train_default.yaml`](configs/train_default.yaml) — датасет, сплит, `DataLoader`, модель, лосс, оптимизатор, планировщик, эпохи, `metric_for_best` (`val_loss` или `val_l2_px`), `grad_clip_norm`, tqdm, размер канваса для L2, каталог чекпойнтов, флаги MLflow.
- **Код:** пакет [`src/tps_dewarp/training/`](src/tps_dewarp/training/) — `load_train_config`, `build_tps_dataloaders`, `build_model`, класс `Trainer`.
- **Запуск из ноутбука:** `dagshub.init` и при необходимости `mlflow.set_experiment` / `mlflow.start_run` — до создания `Trainer`, затем `trainer.fit(resume_from=None)`. Для продолжения: `trainer.fit(resume_from="…/last.pt")` (или `best.pt`); в чекпойнте хранится `mlflow_run_id`, Trainer возобновляет тот же run.
- **Чекпойнты** (в `checkpoint.dir` из YAML): `last.pt` после каждой эпохи, `best.pt` при улучшении метрики из `training.metric_for_best`, `final.pt` в конце успешного `fit`.
- **MLflow:** при улучшении best — опционально `log_artifact` для `best.pt`; в конце — один вызов `mlflow.pytorch.log_model` для финальной модели (без `registered_model_name`). **`mlflow.end_run`** вызывается только если этот же `fit` сам открыл run; если run уже активен (открыт в ноутбуке), Trainer его не закрывает.
