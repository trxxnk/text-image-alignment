import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 difficulty_indices: dict,
                 batch_size: int,
                 weights: dict,
                 random_seed: int = 42):

        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_seed)

        # убираем пустые классы
        self.difficulty_indices = {
            k: np.array(v)
            for k, v in difficulty_indices.items()
            if len(v) > 0
        }

        self.weights = {
            k: weights.get(k, 0.0)
            for k in self.difficulty_indices.keys()
        }

        self.keys = list(self.difficulty_indices.keys())

        total_samples = sum(len(v) for v in self.difficulty_indices.values())
        self.num_batches = total_samples // batch_size

        self._prepare()

    # ===== подготовка распределения =====
    def _prepare(self):
        probs = np.array([self.weights[k] for k in self.keys], dtype=np.float32)

        if probs.sum() == 0:
            probs = np.ones_like(probs)

        probs = probs / probs.sum()

        raw_counts = probs * self.batch_size
        counts = np.floor(raw_counts).astype(int)

        # добиваем до batch_size
        remainder = self.batch_size - counts.sum()

        if remainder > 0:
            frac = raw_counts - counts
            for i in np.argsort(-frac)[:remainder]:
                counts[i] += 1

        self.counts = dict(zip(self.keys, counts))

    # ===== обновление curriculum =====
    def set_weights(self, weights: dict):
        for k in self.keys:
            self.weights[k] = weights.get(k, 0.0)
        self._prepare()

    # ===== генерация батчей =====
    def __iter__(self):

        # создаём "бесконечные" циклы индексов
        pools = {}
        pointers = {}

        for k in self.keys:
            idxs = self.difficulty_indices[k]
            perm = self.rng.permutation(idxs)
            pools[k] = perm
            pointers[k] = 0

        for _ in range(self.num_batches):

            batch = []

            for k in self.keys:
                n = self.counts[k]
                if n == 0:
                    continue

                idxs = pools[k]
                ptr = pointers[k]

                # если не хватает — пересэмплируем
                if ptr + n > len(idxs):
                    idxs = self.rng.permutation(self.difficulty_indices[k])
                    pools[k] = idxs
                    ptr = 0

                selected = idxs[ptr:ptr + n]
                pointers[k] = ptr + n

                batch.extend(selected.tolist())

            # защита (редкий случай)
            if len(batch) != self.batch_size:
                extra = self.batch_size - len(batch)
                fallback = self.rng.choice(
                    np.concatenate(list(self.difficulty_indices.values())),
                    size=extra
                )
                batch.extend(fallback.tolist())

            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
