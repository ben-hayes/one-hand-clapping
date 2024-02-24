# The Sound of One Hand CLAPping

Evolutionary search over VSTi synthesiser parameters using CLAP embeddings as a fitness function.

Root package is called `ohc`.

## Evolutionary Algorithm (Soumya)

Directory: `ohc/search`
Import path: `from ohc.search import ...`

Should provide a wrapper around `evotorch` that allows us to pass in configurations and call `run`. Ideally would implement some audio metrics (FAD? Quality Diversity?) that allow us to get a sense of what's working well.

Creates a population of candidate patches and passes them to the `FitnessFunction`.

Should allow us to easily configure (try out different algorithms and parameters).

```python
EvolutionarySearch.__init__(
    fitness_function: FitnessFunction,
    **config
)
EvolutionarySearch.run(
    [run configuration parameters]
)
```

## VSTi Host (Jordie)

Directory: `ohc/vst`
Import path: `from ohc.vst import ...`

VST host should be able to render audio given a parameter vector, as well as translate to/from parameter vectors. It should expose a list of parameters and allow active parameters to be set.

```python
VstiHost.__init__(
    vst_file: Path,
    inactive_param_behaviour: Literal["random", "fixed"],
)
VstiHost.list_params() -> List[str]
VstiHost.set_active_params(
    active_params: List[str]
)
VstiHost.render(
    params: evotorch.SolutionBatch,
    midi_note: int,
    note_duration_in_seconds: float,
    tail_duration_in_seconds: float,
    callback: Callable[Union[torch.Tensor, np.ndarray], int]  # callback(audio, index_in_batch) -- should be called for each batch item as soon as the output is ready.
)
```

## Fitness Function (Ben)

Directory: `ohc/fitness`
Import path: `from ohc.fitness import ...`

```python
FitnessFunction.__init__(
    vsti_host: VstiHost
    clap_similarity: CLAPSimilarity
)
FitnessFunction.compute(
    solutions: evotorch.SolutionBatch
)

CLAPSimilarity.__init__(
    batch_size: int,
    device: torch.Device,
)
CLAPSimilarity.compute(
    audio: torch.Tensor,
    target_embedding: torch.Tensor
)
CLAPSimilarity.get_text_embedding(
    input: str
)
CLAPSimilarity.get_audio_embedding(
    input: torch.Tensor
)
```

## Interface (Ash)

Should allow:

- upload / selection of VSTi
- listing / selection of parameters
- input of text / Audio prompt
- visualisation of best output, output generation, (and algorithm history? i.e. intermediate results)

Should use entry point defined in `ohc/run.py` (`from ohc import run`).
