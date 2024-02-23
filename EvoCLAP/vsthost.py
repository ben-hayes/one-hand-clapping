# VstiHost.__init__(
#     vst_file: Path,
#     inactive_param_behaviour: Literal["random", "fixed"],
# )
# VstiHost.list_params() -> List[str]
# VstiHost.set_active_params(
#     active_params: List[str]
# )
# VstiHost.params_to_solution(
#     params: dict[str, Any],
# ) -> torch.Tensor
# VstiHost.solution_to_params(
#     solution: evotorch.Solution
# ) -> dict[str, Any]
# VstiHost.render(
#     params: Union[dict[str, Any], evotorch.Solution]  # (input type as you see fit!)
#     individual_idx: int,
#     midi_note: int,
#     note_duration_in_seconds: float,
#     tail_duration_in_seconds: float,
# ) -> torch.Tensor  # audio output (2, T)

class VstiHost:
    def __init__(self,
                 vst_file: str,
                 inactive_param_behaviour: str = "random",
                 ):
        self.vst_file = vst_file
        self.inactive_param_behaviour = inactive_param_behaviour

