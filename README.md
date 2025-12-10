# BipedalWalker-v3 RL Algorithms Comparison

Проект по сравнению алгоритмов глубокого обучения с подкреплением на задаче BipedalWalker-v3.

## Обзор

Натренированы и сравнены три алгоритма:
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed DDPG)

## Результаты

### Лучшие модели (Best Models):

1. **PPO** - 295.87 ± 1.81 ⭐⭐⭐
   - Лучший и самый стабильный результат
   - 100% успешных эпизодов (>200 reward)
   - Модель: `models/ppo_20251209_223634/best/best_model`

2. **TD3** - 257.49 ± 60.69 ⭐⭐
   - Хороший результат, но менее стабильный
   - 90% успешных эпизодов
   - Модель: `models/td3_20251209_223718/best/best_model`

3. **SAC** - -9.73 ± 4.03 ✗
   - Не научился эффективной политике
   - 0% успешных эпизодов
   - Модель: `models/sac_20251209_223638/best/best_model`

### Final модели:

- **TD3** - 277.23 ± 2.74 (самый стабильный)
- **PPO** - 243.60 ± 103.72 (нестабильный)
- **SAC** - -44.74 ± 6.69 (плохо)

## Структура проекта

```
advai-final/
├── models/                      # Натренированные модели
│   ├── ppo_20251209_223634/
│   ├── sac_20251209_223638/
│   └── td3_20251209_223718/
├── newresults/                  # Результаты оценки
│   ├── comprehensive_comparison.png  # Сводный график
│   ├── final_report.txt             # Текстовый отчет
│   └── evaluation_*.json            # JSON результаты
├── videos/                      # Видео лучших моделей
│   ├── ppo_best_model.mp4
│   ├── sac_best_model.mp4
│   └── td3_best_model.mp4
├── train.py                     # Тренировка одного алгоритма
├── train_all.py                 # Тренировка всех алгоритмов
├── eval.py                      # Оценка одной модели
├── eval_all.py                  # Оценка всех моделей
├── plot.py                      # Базовая визуализация
├── generate_report.py           # Генерация отчета
└── record_videos.py             # Запись видео
```

## Использование

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Тренировка

Тренировка одного алгоритма:
```bash
python train.py --algorithm PPO --timesteps 500000
```

Тренировка всех алгоритмов:
```bash
python train_all.py
```

### 3. Оценка

Оценка одной модели:
```bash
python eval.py --model-path models/ppo_20251209_223634/best/best_model --algorithm PPO --n-episodes 10
```

Оценка всех моделей (final):
```bash
python eval_all.py --n-episodes 10
```

Оценка всех моделей (best):
```bash
python eval_all.py --n-episodes 10 --use-best
```

С записью видео:
```bash
python eval_all.py --n-episodes 10 --use-best --save-videos
```

### 4. Генерация отчета

```bash
python generate_report.py
```

Создает:
- `newresults/comprehensive_comparison.png` - сводный график с 4 визуализациями
- `newresults/final_report.txt` - текстовый отчет с анализом

### 5. Запись видео

```bash
python record_videos.py
```

Записывает видео всех best моделей в папку `videos/`.

## Анализ результатов

### Почему PPO лучше?

1. **Стабильность**: PPO best показал наименьшую дисперсию (σ = 1.81)
2. **Надежность**: 100% успешных эпизодов
3. **Производительность**: Средний reward близок к максимуму (295.87 из ~300)

### Почему SAC провалился?

1. Отрицательные rewards во всех эпизодах
2. Модель не научилась удерживать равновесие
3. Возможные причины:
   - Неподходящие гиперпараметры
   - Недостаточное количество timesteps
   - SAC может требовать больше exploration

### TD3 vs PPO

- **TD3 final**: Более стабильный (σ = 2.74 vs 103.72)
- **PPO best**: Более высокий результат (295.87 vs 257.49)
- **Вывод**: PPO лучше находит оптимальную политику, TD3 более консервативен

## Визуализация

### Comprehensive Comparison Plot включает:

1. **Final vs Best Models** - сравнение финальных и лучших моделей
2. **Stability Comparison** - сравнение стабильности (стандартное отклонение)
3. **Reward Distribution** - распределение наград по эпизодам
4. **Performance Summary** - таблица с итогами

## Файлы результатов

- `newresults/evaluation_*.json` - детальные результаты оценки
- `newresults/comparison_*.png` - графики сравнения
- `newresults/comprehensive_comparison.png` - сводный график
- `newresults/final_report.txt` - текстовый отчет

## Видео

В папке `videos/` находятся записи лучших моделей:
- `ppo_best_model.mp4` - отличная походка
- `td3_best_model.mp4` - хорошая походка
- `sac_best_model.mp4` - модель падает

## Выводы

1. **PPO** показал себя как лучший алгоритм для BipedalWalker-v3
2. **TD3** - хорошая альтернатива с более стабильным обучением
3. **SAC** требует дополнительной настройки для этой задачи

## Рекомендации

Для будущих экспериментов:
1. Увеличить timesteps для SAC (1M+)
2. Провести grid search по гиперпараметрам
3. Попробовать другие алгоритмы (TRPO, A2C, DQN)
4. Добавить curriculum learning
