from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from torchmetrics import Metric
import torch


class RMSEMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: preds {preds.shape} vs target {target.shape}"
            )

        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.sum_squared_error / self.total)


METRIC_DICT = {
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "rmse": RMSEMetric,
    "r2": R2Score,
    "pearson": PearsonCorrCoef,
    "spearman": SpearmanCorrCoef,
}

if __name__ == "__main__":
    metric = RMSEMetric()
    print(metric)