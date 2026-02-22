import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from datetime import datetime


def save_lunar_model_onnx(
    model: nn.Module,
    path: Union[str, Path],
    *,
    continuous: bool = False,
    opset: int = 18,
    include_timestamp: bool = True,
) -> Path:
    """
    Export a trained Lunar Lander model to ONNX format (inference only).

    - Automatically generates example input of shape [1, 8].
    - Adds timestamp (YYYY-MM-DD_HH-MM) before extension to avoid overwrite.
    - Uses the new PyTorch dynamo ONNX exporter.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    path : str | Path
        Base filename (e.g., "output/policy.onnx"). ".onnx" enforced.
    continuous : bool, optional
        True if the model is continuous (2D action), False for discrete (4D Q-values).
    opset : int, optional
        ONNX opset version (default 18).
    include_timestamp : bool, optional
        Whether to append timestamp to filename. Default True.

    Returns
    -------
    Path
        Final ONNX file path.
    """
    model = model.to("cpu").eval()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .onnx extension
    base = path.with_suffix(".onnx")

    # Add timestamp
    if include_timestamp:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_path = base.with_name(f"{base.stem}_{ts}{base.suffix}")
    else:
        save_path = base

    # Lunar Lander observation dimension = 8
    dummy_input = torch.zeros(1, 8, dtype=torch.float32)

    input_names = ["state"]
    output_name = "action" if continuous else "q_values"

    # ✅ Proper dynamic_shapes format for new exporter
    dynamic_shapes = ({0: torch.export.Dim("batch")},)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),  # tuple required by dynamo exporter
            save_path.as_posix(),
            export_params=True,
            opset_version=opset,
            input_names=input_names,
            output_names=[output_name],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )

    print(f"✅ Lunar Lander model exported to {save_path.resolve()}")
    return save_path


def save_model_pytorch(model: nn.Module, path: Union[str, Path]) -> Path:
    """
    Save only the model weights (suitable for inference use).
    Appends a human-readable timestamp (YYYY-MM-DD_HH-MM) to the filename
    so that files are not overwritten.

    Parameters
    ----------
    model : nn.Module
        The model to save.
    path : str or Path
        Base path for saving (e.g., "output/target_q.pth").
        A timestamp will be added before the extension.

    Returns
    -------
    Path
        The actual file path used for saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Format timestamp (e.g., 2025-09-13_14-35)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Insert timestamp before extension
    if path.suffix:
        save_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    else:
        save_path = path.with_name(f"{path.stem}_{timestamp}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path.resolve()}")
