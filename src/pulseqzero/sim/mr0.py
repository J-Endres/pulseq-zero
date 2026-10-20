import MRzeroCore
import torch


class MR0:
    """MR-zero's PDG simulation of a sequence on ``data`` (a ``SimData``)."""

    def __init__(
        self,
        data: MRzeroCore.SimData,
        *,
        print_progress: bool = False,
        graph_kwargs: dict | None = None,
        execute_kwargs: dict | None = None,
    ):
        """``graph_kwargs`` go to ``compute_graph`` (``max_state_count``,
        ``min_state_mag``, ...), ``execute_kwargs`` to ``execute_graph``."""
        self.data = data
        self.print_progress = print_progress
        self.graph_kwargs = graph_kwargs or {}
        self.execute_kwargs = execute_kwargs or {}

    def simulate(self, seq: MRzeroCore.Sequence) -> torch.Tensor:
        graph = MRzeroCore.compute_graph(seq, self.data, **self.graph_kwargs)
        return MRzeroCore.execute_graph(
            graph,
            seq,
            self.data,
            print_progress=self.print_progress,
            **self.execute_kwargs,
        )
