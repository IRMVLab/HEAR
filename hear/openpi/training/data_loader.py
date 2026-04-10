from collections import OrderedDict
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn], audio_history_window: int = 0):
        if isinstance(dataset, tuple):
            self._dataset = dataset[0]
        else:
            self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self.audio_history_window = audio_history_window
        self._audio_cache: OrderedDict[int, typing.Any] = OrderedDict()
        if self.audio_history_window > 0:
            self._audio_cache_max_size = min(4096, max(16, self.audio_history_window * 16))
        else:
            self._audio_cache_max_size = 0

    def _cache_audio(self, index: int, audio: typing.Any) -> typing.Any:
        if self._audio_cache_max_size <= 0:
            return audio
        self._audio_cache[index] = audio
        self._audio_cache.move_to_end(index)
        if len(self._audio_cache) > self._audio_cache_max_size:
            self._audio_cache.popitem(last=False)
        return audio

    def _get_audio(self, index: int) -> typing.Any:
        if self._audio_cache_max_size <= 0:
            return self._dataset[index]["observation.audio"]
        if index in self._audio_cache:
            audio = self._audio_cache[index]
            self._audio_cache.move_to_end(index)
            return audio
        audio = self._dataset[index]["observation.audio"]
        self._audio_cache[index] = audio
        if len(self._audio_cache) > self._audio_cache_max_size:
            self._audio_cache.popitem(last=False)
        return audio

    def _hist_audio_matches_window(self, hist_audio: typing.Any) -> bool:
        if hist_audio is None:
            return False
        try:
            return int(hist_audio.shape[0]) == self.audio_history_window
        except Exception:
            return False

    def _build_hist_audio(self, index: int, frame_index: int, curr_audio: typing.Any) -> torch.Tensor:
        base_index = index - frame_index
        start_frame = frame_index - (self.audio_history_window - 1)
        pad = 0
        if start_frame < 0:
            pad = -start_frame
            start_frame = 0

        hist_audio: list[typing.Any] = []
        if pad:
            base_audio = self._get_audio(base_index)
            hist_audio.extend([base_audio] * pad)

        for frame in range(start_frame, frame_index + 1):
            hist_index = base_index + frame
            if curr_audio is not None and hist_index == index:
                hist_audio.append(self._cache_audio(hist_index, curr_audio))
            else:
                hist_audio.append(self._get_audio(hist_index))
        return torch.stack(hist_audio, dim=0)

    def _is_same_episode(self, curr_sample: dict, next_sample: dict, frame_index: int) -> bool:
        curr_episode = curr_sample.get("episode_index")
        next_episode = next_sample.get("episode_index")
        if curr_episode is not None and next_episode is not None:
            try:
                return int(curr_episode) == int(next_episode)
            except Exception:
                return curr_episode == next_episode

        next_frame_index = next_sample.get("frame_index")
        if next_frame_index is None:
            return False
        try:
            return int(next_frame_index) == frame_index + 1
        except Exception:
            return False

    def _build_next_audio(
        self,
        index: int,
        frame_index: int,
        curr_sample: dict,
        curr_audio: typing.Any,
    ) -> typing.Any:
        if curr_audio is None:
            return None
        next_audio = curr_sample.get("observation.next_audio")
        if next_audio is not None:
            return next_audio
        next_index = index + 1
        if next_index >= len(self._dataset):
            return curr_audio
        next_sample = self._dataset[next_index]
        if not self._is_same_episode(curr_sample, next_sample, frame_index):
            return curr_audio
        next_audio = next_sample.get("observation.audio")
        if next_audio is None:
            return curr_audio
        return self._cache_audio(next_index, next_audio)

    def __getitem__(self, index: SupportsIndex) -> T_co:

        curr_sample = self._dataset[index]
        frame_index = curr_sample["frame_index"]
        index_value = index.__index__()
        frame_index_value = int(frame_index)
        curr_audio = curr_sample.get("observation.audio")


        if self.audio_history_window > 0:
            hist_audio = curr_sample.get("observation.hist_audio")
            if not self._hist_audio_matches_window(hist_audio):
                if curr_audio is not None:
                    self._cache_audio(index_value, curr_audio)
                hist_audio = self._build_hist_audio(index_value, frame_index_value, curr_audio)
            curr_sample["observation.hist_audio"] = hist_audio
        else:
            curr_sample["observation.hist_audio"] = None
        curr_sample["observation.next_audio"] = self._build_next_audio(
            index_value, frame_index_value, curr_sample, curr_audio
        )

        return self._transform(curr_sample)

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig, audio_history_window: int = 0,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)], audio_history_window=audio_history_window)
    episode_meta = dataset_meta.episodes
    return dataset, episode_meta


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False, audio_history_window: int = 0,) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        audio_history_window=audio_history_window,
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )

def validate_sampling_config(config: Dict[Tuple[float, float], float]) -> Dict[Tuple[float, float], float]:

    if not config:
        return config


    sorted_intervals = sorted(config.keys(), key=lambda x: x[0])


    total_prob = sum(config.values())
    if not np.isclose(total_prob, 1.0, atol=1e-3):
        logging.warning(f"Sampling config probabilities sum to {total_prob:.4f}, expected 1.0. "
                        "The sampler will normalize this, but ensure this is intentional.")


    prev_end = 0.0
    for start, end in sorted_intervals:
        if start >= end:
            raise ValueError(f"Invalid interval ({start}, {end}): start must be < end.")

        if start < prev_end:

            raise ValueError(f"Sampling intervals overlap! Overlap detected between previous end {prev_end} and current start {start}.")

        if start > prev_end:

            logging.warning(f"Gap detected in sampling intervals: {prev_end:.2f} to {start:.2f}. "
                            "Data in this range will have 0 probability (NEVER sampled).")

        prev_end = end

    if prev_end < 1.0:
        logging.warning(f"Sampling intervals end at {prev_end:.2f}, not 1.0. "
                        "Data from {prev_end:.2f} to 1.0 will NOT be sampled.")

    return dict(sorted(config.items()))

def compute_sample_weights(
    episode_meta: dict,
    sampling_config: Dict[Tuple[float, float], float],
    total_dataset_len: int
) -> torch.Tensor:


    sampling_config = validate_sampling_config(sampling_config)


    density_weights = {}
    for (start_r, end_r), target_prob in sampling_config.items():
        interval_len = end_r - start_r

        if interval_len > 0:
            density_weights[(start_r, end_r)] = target_prob / interval_len
        else:
             density_weights[(start_r, end_r)] = 0.0


    all_weights = np.zeros(total_dataset_len, dtype=np.float64)

    current_idx = 0


    for _, meta in episode_meta.items():
        ep_len = meta['length']
        ep_start_global = current_idx


        for (start_r, end_r), weight_density in density_weights.items():

            local_start = int(np.floor(start_r * ep_len))
            local_end = int(np.ceil(end_r * ep_len))


            local_start = max(0, min(local_start, ep_len))
            local_end = max(0, min(local_end, ep_len))

            if local_end > local_start:

                all_weights[ep_start_global + local_start : ep_start_global + local_end] = weight_density

        current_idx += ep_len


    if current_idx != total_dataset_len:
        if current_idx < total_dataset_len:
             all_weights = np.pad(all_weights, (0, total_dataset_len - current_idx), constant_values=0.0)
        else:
             all_weights = all_weights[:total_dataset_len]

    return torch.as_tensor(all_weights, dtype=torch.double)

def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    sampling_weights_config: Optional[Dict[Tuple[float, float], float]] = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        sampling_weights_config=sampling_weights_config,
        audio_history_window=config.audio_history_window,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    sampling_weights_config: Dict[Tuple[float, float], float] | None = None,
    audio_history_window: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:


    dataset, episode_meta = create_torch_dataset(data_config, action_horizon, model_config, audio_history_window)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, audio_history_window=audio_history_window)

    world_size = 1
    rank = 0
    if framework == "pytorch" and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()


    if framework == "pytorch":
        local_batch_size = batch_size // world_size
    else:
        local_batch_size = batch_size // jax.process_count()

    sampler = None


    if framework == "pytorch":

        if sampling_weights_config is not None and shuffle:
            logging.info(f"[Rank {rank}] Building WeightedRandomSampler with config: {sampling_weights_config}")


            weights = compute_sample_weights(episode_meta, sampling_weights_config, len(dataset))


            num_samples = len(dataset) // world_size


            generator = torch.Generator()
            generator.manual_seed(seed + rank)

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True,
                generator=generator
            )

        elif torch.distributed.is_initialized():

            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True,
                seed=seed,
            )


    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,

        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,


        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1

                def _maybe_tensor(x):
                    if (isinstance(x, (np.ndarray, jax.Array, torch.Tensor)) or np.isscalar(x)) and not isinstance(x, str):
                        return torch.as_tensor(x)
                    return x

                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(_maybe_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    def _stack_or_list(*xs):
        first = xs[0]
        if isinstance(first, str):
            return list(xs)
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return jax.tree.map(_stack_or_list, *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
